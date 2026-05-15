import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# 如果没有装 open_clip_torch:
# pip install open_clip_torch
try:
    import open_clip
except ImportError as e:
    raise SystemExit(
        "open_clip_torch not installed. Install with:\n"
        "  pip install open_clip_torch"
    ) from e


class LGGSNPairsDataset(Dataset):
    """基于 logs/lggsn_pairs.csv 的简单 Dataset"""

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        required_cols = [
            "query", "x", "y", "z",
            "roll", "pitch", "yaw",
            "width", "obj_height", "label",
        ]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        self.texts = df["query"].astype(str).tolist()
        geom_cols = ["x", "y", "z", "roll", "pitch", "yaw", "width", "obj_height"]
        self.geom = torch.tensor(df[geom_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.geom[idx], self.labels[idx]


class LGGSN(nn.Module):
    """玩具版 LG-GSN：text_emb + pose_emb -> match score"""

    def __init__(self, text_dim=512, geom_dim=8, hidden_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个 logit
        )

    def forward(self, text_feat, geom):
        t = self.text_proj(text_feat)
        g = self.geom_proj(geom)
        h = torch.cat([t, g], dim=-1)
        logit = self.classifier(h).squeeze(-1)
        return logit


def build_loaders(csv_path: Path, batch_size: int = 16, val_ratio: float = 0.2):
    dataset = LGGSNPairsDataset(csv_path)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def train(args):
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "logs" / "lggsn_pairs.csv"
    if args.csv is not None:
        csv_path = Path(args.csv)

    print("[INFO] Using pairs CSV:", csv_path)
    train_loader, val_loader = build_loaders(csv_path, args.batch_size, args.val_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    print("[INFO] Loading CLIP text encoder (ViT-B-32, laion2b_s34b_b79k)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 冻结 CLIP 参数
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    text_dim = clip_model.text_projection.shape[1]
    model = LGGSN(text_dim=text_dim, geom_dim=8, hidden_dim=args.hidden_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def encode_text_batch(text_list):
        with torch.no_grad():
            tokens = tokenizer(text_list).to(device)
            text_feat = clip_model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def run_epoch(loader, train_mode=True):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for texts, geom, labels in loader:
            geom = geom.to(device)
            labels = labels.to(device)

            text_feats = encode_text_batch(texts)

            logits = model(text_feats, geom)
            loss = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        acc = total_correct / max(1, total)
        return avg_loss, acc

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc = run_epoch(val_loader, train_mode=False)

        print(
            f"[EP {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_dir = root / "logs"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / "lggsn_baseline.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "text_dim": text_dim,
                    "geom_dim": 8,
                    "hidden_dim": args.hidden_dim,
                },
                out_path,
            )
            print(f"[SAVE] New best model saved to {out_path}, val_acc={val_acc:.3f}")

    print("[DONE] Training finished. Best val_acc =", best_val_acc)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline LG-GSN training on lggsn_pairs.csv"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to lggsn_pairs.csv (default: logs/lggsn_pairs.csv)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
