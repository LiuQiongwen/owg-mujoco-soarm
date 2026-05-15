#!/usr/bin/env python3
"""
Train a pure-numpy MLP world-model predictor from collected transition data.

Architecture
------------
Input  (22-dim): grasp_pose(6) + obj_pos(3) + obj_quat(4) + pc_stats(9)
Three independent heads:
  success_prob  — binary classifier (BCE loss)
  dz_pred       — regressor (MSE loss)
  fell_off_prob — binary classifier (BCE loss)

All heads share the same scaler but have separate weights.
Model is saved as world_model/mlp_predictor.pkl (pure numpy — no sklearn).

Usage:
  MUJOCO_GL=egl conda run -n bridge python world_model/train_mlp_predictor.py
  MUJOCO_GL=egl conda run -n bridge python world_model/train_mlp_predictor.py \\
    --data-dir data/transitions --epochs 400
  MUJOCO_GL=egl conda run -n bridge python world_model/train_mlp_predictor.py --eval
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.transition_logger import TransitionLogger, TRANSITIONS_DIR, FEATURE_DIM

MODEL_PATH = Path("world_model/mlp_predictor.pkl")
_HIDDEN    = (64, 64)


# ── Pure-numpy MLP ────────────────────────────────────────────────────────────

class _Scaler:
    """Feature-wise z-score normalisation."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mu_    = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-8
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sigma_


class _MLP:
    """
    2-hidden-layer MLP with ReLU activations, mini-batch SGD.

    task = "classify" → sigmoid output + BCE loss
    task = "regress"  → linear output  + MSE loss
    """

    def __init__(self, in_dim: int, hidden: tuple = _HIDDEN, out_dim: int = 1,
                 task: str = "classify", lr: float = 3e-3,
                 batch: int = 32, epochs: int = 300, seed: int = 0):
        rng   = np.random.default_rng(seed)
        sizes = [in_dim, *hidden, out_dim]
        self.W = [rng.normal(0, np.sqrt(2 / n), (n, m)).astype(np.float64)
                  for n, m in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(m, dtype=np.float64) for m in sizes[1:]]
        self.task   = task
        self.lr     = lr
        self.batch  = batch
        self.epochs = epochs

    # ── forward ──────────────────────────────────────────────────────────────

    def _fwd(self, X: np.ndarray) -> list:
        """Return list of activations (including input layer)."""
        acts = [X.astype(np.float64)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            acts.append(h)
        return acts

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "_MLP":  # noqa: E501
        X = X.astype(np.float64)
        y = y.reshape(len(X), -1).astype(np.float64)
        n = len(X)
        rng = np.random.default_rng(0)

        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):
                bi  = idx[s : s + self.batch]
                Xb, yb = X[bi], y[bi]
                acts = self._fwd(Xb)
                out  = acts[-1]

                # output gradient
                if self.task == "classify":
                    p = 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))
                    g = (p - yb) / len(bi)
                else:
                    g = 2.0 * (out - yb) / len(bi)

                # backprop
                for i in range(len(self.W) - 1, -1, -1):
                    dW = acts[i].T @ g
                    db = g.sum(axis=0)
                    if i > 0:
                        g = (g @ self.W[i].T) * (acts[i] > 0)
                    self.W[i] -= self.lr * dW
                    self.b[i] -= self.lr * db

            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                acts  = self._fwd(X)
                out   = acts[-1]
                if self.task == "classify":
                    p    = 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))
                    loss = -np.mean(y * np.log(p + 1e-7) +
                                    (1 - y) * np.log(1 - p + 1e-7))
                else:
                    loss = float(np.mean((out - y) ** 2))
                print(f"    epoch {epoch:>3d}  loss={loss:.4f}")

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Class-1 probability for classifiers."""
        out = self._fwd(X.astype(np.float64))[-1]
        return (1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Regression prediction (or thresholded class for classifiers)."""
        out = self._fwd(X.astype(np.float64))[-1].ravel()
        if self.task == "classify":
            return (out > 0).astype(int)
        return out


# ── Metrics (pure numpy) ──────────────────────────────────────────────────────

def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order   = np.argsort(-y_score)
    y_true  = y_true[order]
    n_pos   = y_true.sum()
    n_neg   = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.cumsum(y_true)     / n_pos
    fpr = np.cumsum(1 - y_true) / n_neg
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ── Train / eval / save ───────────────────────────────────────────────────────

def train_all(X: np.ndarray, y: np.ndarray,
              epochs: int = 300, verbose: bool = True) -> dict:
    """
    Fit scaler + three MLP heads.

    Returns a model dict consumable by save_model / load_model.
    """
    y_success = y[:, 0]
    y_dz      = y[:, 1]
    y_fell    = y[:, 2]

    if verbose:
        print(f"  samples={len(X)}  features={X.shape[1]}  epochs={epochs}")
        print(f"  success_rate={y_success.mean():.2f}  "
              f"fell_off_rate={y_fell.mean():.2f}  "
              f"dz_mean={y_dz.mean():.3f}")

    scaler = _Scaler()
    X_sc   = scaler.fit_transform(X)

    in_dim = X_sc.shape[1]

    if verbose:
        print("  training clf_success …")
    clf_success = _MLP(in_dim, task="classify", epochs=epochs, seed=0
                       ).fit(X_sc, y_success, verbose=verbose)

    if verbose:
        print("  training reg_dz …")
    reg_dz = _MLP(in_dim, task="regress", epochs=epochs, seed=1,
                  lr=1e-3).fit(X_sc, y_dz, verbose=verbose)

    if verbose:
        print("  training clf_fell …")
    clf_fell = _MLP(in_dim, task="classify", epochs=epochs, seed=2,
                    ).fit(X_sc, y_fell, verbose=verbose)

    return {"scaler": scaler, "clf_success": clf_success,
            "reg_dz": reg_dz, "clf_fell": clf_fell}


def evaluate(model: dict, X: np.ndarray, y: np.ndarray,
             verbose: bool = True) -> dict:
    X_sc       = model["scaler"].transform(X)
    y_success  = y[:, 0]
    y_dz       = y[:, 1]
    y_fell     = y[:, 2]

    p_success  = model["clf_success"].predict_prob(X_sc)
    dz_pred    = model["reg_dz"].predict(X_sc)
    p_fell     = model["clf_fell"].predict_prob(X_sc)

    metrics = {
        "n":            len(X),
        "success_auc":  _roc_auc(y_success, p_success),
        "dz_mae":       _mae(y_dz, dz_pred),
        "fell_auc":     _roc_auc(y_fell, p_fell),
    }
    if verbose:
        print("\n  Eval (test set):")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return metrics


def save_model(model: dict, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved → {path}")


class _PortableUnpickler(pickle.Unpickler):
    """Redirect __main__ class lookups to this module so the model loads
    regardless of whether it was saved from a script or an import context."""
    def find_class(self, module, name):
        if module == "__main__":
            import world_model.train_mlp_predictor as _mod
            return getattr(_mod, name)
        return super().find_class(module, name)


def load_model(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return _PortableUnpickler(f).load()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir",    default=str(TRANSITIONS_DIR))
    ap.add_argument("--model-path",  default=str(MODEL_PATH))
    ap.add_argument("--epochs",      type=int, default=300)
    ap.add_argument("--test-size",   type=float, default=0.2)
    ap.add_argument("--eval",        action="store_true",
                    help="Evaluate an existing model instead of re-training")
    ap.add_argument("--quiet",       action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    logger  = TransitionLogger(out_dir=Path(args.data_dir))
    X, y, _ = logger.load_dataset()

    if len(X) == 0:
        print("[ERROR] No transition data found. "
              "Run collect_mujoco_transitions.py first.")
        sys.exit(1)

    print(f"Loaded {len(X)} transitions from {args.data_dir}")

    if args.eval:
        model = load_model(Path(args.model_path))
        evaluate(model, X, y, verbose=verbose)
        return

    # Train / test split (shuffle)
    idx = np.random.default_rng(42).permutation(len(X))
    n_test = max(1, int(len(X) * args.test_size)) if len(X) >= 10 else 0
    te, tr = idx[:n_test], idx[n_test:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    print(f"\n=== Training (n_train={len(X_tr)}  n_test={len(X_te)}) ===")
    model = train_all(X_tr, y_tr, epochs=args.epochs, verbose=verbose)

    if len(X_te) >= 2:
        evaluate(model, X_te, y_te, verbose=verbose)

    save_model(model, Path(args.model_path))


if __name__ == "__main__":
    main()
