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
from sklearn.preprocessing import MinMaxScaler

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH    = 'data.csv'
NII_DIR     = 'nii_data'
SAVE_PATH   = 'best_model_float.pt'
TARGET_SIZE = (96, 96, 96)
BATCH_SIZE  = 4
NUM_EPOCHS  = 1_000_000
LR          = 1e-4
PATIENCE    = 20
GPU_IDS     = list(range(8))
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
        img      = torch.from_numpy(img).unsqueeze(0)
        score    = torch.tensor(float(row['score_norm']), dtype=torch.float32)
        return img, score

    def _resize(self, img):
        factors = [t / s for t, s in zip(self.target_size, img.shape[:3])]
        return zoom(img, factors, order=1)

    def _normalize(self, img):
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        return img


class VGG16_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.fc1     = nn.Sequential(
            nn.Linear(512 * 8, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.fc2     = nn.Sequential(
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.head    = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x)).squeeze(1)


def train():
    df = pd.read_csv(CSV_PATH)

    # MinMax normalize score to [0, 1]
    scaler         = MinMaxScaler()
    df['score_norm'] = scaler.fit_transform(df[['score']]).squeeze()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

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

    model = VGG16_3D()
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    grad_scaler = GradScaler()

    best_val_loss   = float('inf')
    patience_count  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, scores in train_loader:
            imgs, scores = imgs.to(device), scores.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss  = criterion(preds, scores)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, scores in val_loader:
                imgs, scores = imgs.to(device), scores.to(device)
                with autocast():
                    preds    = model(imgs)
                    val_loss += criterion(preds, scores).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(
            f'Epoch [{epoch:3d}/{NUM_EPOCHS}] '
            f'Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}'
        )

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            patience_count = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, SAVE_PATH)
            print(f'  -> Saved best model  (val_mse={avg_val:.6f})')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch} epochs (patience={PATIENCE})')
                break

    print(f'\nTraining complete. Best val MSE: {best_val_loss:.6f}')


if __name__ == '__main__':
    train()
