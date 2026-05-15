import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from lggsn_model import LGGSN  # 仍然复用你原来的模型

CSV_PATH   = os.environ.get("LGGSN_CSV",    "grasp_6dof/dataset/all_lggsn.csv")
CKPT_PATH  = os.environ.get("LGGSN_CKPT",   "grasp_6dof/models/lggsn_geom_only.pt")
N_EPOCHS   = int(os.environ.get("LGGSN_EPOCHS", 20))


class LGGSNDataset(Dataset):
    """
    几何 + 质量特征 → 抓取好坏(label) 的数据集。

    这里不再使用真实 query，而是给一个固定的 dummy query id = 0，
    这样可以沿用 LGGSN(geom, query_id) 这个接口。
    """
    def __init__(self, df, feature_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        geom = torch.tensor(
            row[self.feature_cols].values.astype("float32"),
            dtype=torch.float32,
        )
        # 单一虚拟 query：全是 0
        q_id = torch.tensor(0, dtype=torch.long)

        # label 列：0 = bad grasp, 1 = good grasp
        y = torch.tensor(float(row["label"]), dtype=torch.float32)

        return geom, q_id, y


def main():
    # 1) 读取 CSV
    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        raise ValueError(
            "CSV 里找不到 'label' 列，请确认 convert_to_lggsn.py 已经生成 label。"
        )

    print("Label distribution from CSV:", df["label"].value_counts().to_dict())

    # 几何 + 质量特征列，根据你的 all_lggsn.csv 来选
    feature_cols = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "width", "score",
        "dz", "dz_lift", "need_dz", "H",
    ]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"CSV 里缺少特征列: {c}")

    # 2) 判断能不能做分层划分
    class_counts = df["label"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        print("[WARN] 至少有一个类别样本数 < 2，不能做 stratified split，将不使用 stratify。")
        stratify = None
    else:
        stratify = df["label"]

    # 3) 按行拆分 train / val，然后构建 Dataset
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_ds = LGGSNDataset(train_df, feature_cols)
    val_ds = LGGSNDataset(val_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 4) 构建模型
    # 我们只有 1 个 dummy query，所以 n_queries = 1
    model = LGGSN(
        n_queries=1,
        geom_dim=len(feature_cols),  # 根据当前特征列自动设置
        query_dim=0,                 # 先不用 query embedding
        hidden_dim=40,               # 你也可以改大一点，比如 64
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss, total_correct, total_n = 0.0, 0, 0

        for geom, q_id, y in loader:
            geom = geom.to(device)
            q_id = q_id.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(train):
                logit = model(geom, q_id).view(-1)  # [B] or [B,1] → [B]
                loss = criterion(logit, y)
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            total_loss += loss.item() * y.size(0)
            pred = (torch.sigmoid(logit) > 0.5).float()
            total_correct += (pred == y).sum().item()
            total_n += y.size(0)

        avg_loss = total_loss / max(total_n, 1)
        avg_acc = total_correct / max(total_n, 1)
        return avg_loss, avg_acc

    # 5) 训练若干个 epoch
    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.2f} | "
            f"val  loss={va_loss:.4f} acc={va_acc:.2f}"
        )
    # 训练结束后保存模型权重
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Saved LGGSN checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()

