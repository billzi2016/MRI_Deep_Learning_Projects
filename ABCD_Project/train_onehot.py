import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH    = 'data.csv'
NII_DIR     = 'nii_data'
SAVE_PATH   = 'best_model_onehot.pt'
TARGET_SIZE = (96, 96, 96)
BATCH_SIZE  = 4
NUM_EPOCHS  = 1_000_000
LR          = 1e-4
PATIENCE    = 20
GPU_IDS     = list(range(8))          # 0-7  H100
NUM_WORKERS = 16
# ────────────────────────────────────────────────────────────────────────


class MRIDataset(Dataset):
    def __init__(self, df, nii_dir, target_size=TARGET_SIZE):
        self.df          = df.reset_index(drop=True)
        self.nii_dir     = nii_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        nii_path = os.path.join(self.nii_dir, row['nii.gz_name'])
        img      = nib.load(nii_path).get_fdata(dtype=np.float32)
        img      = self._resize(img)
        img      = self._normalize(img)
        img      = torch.from_numpy(img).unsqueeze(0)   # (1, D, H, W)
        label    = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label

    def _resize(self, img):
        factors = [t / s for t, s in zip(self.target_size, img.shape[:3])]
        return zoom(img, factors, order=1)

    def _normalize(self, img):
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        return img


class VGG16_3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            # Block 2
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            # Block 3
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            # Block 4
            nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            # Block 5
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))   # → 512 * 8 = 4096
        self.fc1     = nn.Sequential(
            nn.Linear(512 * 8, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.fc2     = nn.Sequential(
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.head    = nn.Linear(4096, num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))


def train():
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    train_loader = DataLoader(
        MRIDataset(train_df, NII_DIR),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        MRIDataset(val_df, NII_DIR),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available = min(len(GPU_IDS), torch.cuda.device_count())
    gpu_ids   = GPU_IDS[:available]

    model = VGG16_3D(num_classes=2)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler      = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler         = GradScaler()

    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                val_loss    += loss.item()
                val_correct += logits.argmax(1).eq(labels).sum().item()
                val_total   += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        print(
            f'Epoch [{epoch:3d}/{NUM_EPOCHS}] '
            f'Train Loss: {train_loss/len(train_loader):.4f}  Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss/len(val_loader):.4f}  Acc: {val_acc:.4f}'
        )

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, SAVE_PATH)
            print(f'  -> Saved best model  (val_acc={val_acc:.4f})')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch} epochs (patience={PATIENCE})')
                break

    print(f'\nTraining complete. Best val acc: {best_val_acc:.4f}')


if __name__ == '__main__':
    train()
