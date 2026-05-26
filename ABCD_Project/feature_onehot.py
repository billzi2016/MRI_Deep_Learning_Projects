import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import zoom

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH    = 'data.csv'
NII_DIR     = 'nii_data'
MODEL_PATH  = 'best_model_onehot.pt'
SAVE_FEAT   = 'features_onehot.csv'
TARGET_SIZE = (96, 96, 96)
BATCH_SIZE  = 4
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
        return img, row['nii.gz_name']

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


def extract():
    df     = pd.read_csv(CSV_PATH)
    loader = DataLoader(
        MRIDataset(df, NII_DIR),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available = min(len(GPU_IDS), torch.cuda.device_count())
    gpu_ids   = GPU_IDS[:available]

    model = VGG16_3D(num_classes=2)
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    model.eval()

    all_feats = []
    all_names = []

    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(device)
            with autocast():
                if isinstance(model, nn.DataParallel):
                    feats = model.module.forward_features(imgs)
                else:
                    feats = model.forward_features(imgs)
            all_feats.append(feats.cpu().numpy())
            all_names.extend(names)
            print(f'  processed {len(all_names)}/{len(df)} samples')

    feats_np = np.concatenate(all_feats, axis=0)   # (N, 4096)
    col_names = [f'feat_{i}' for i in range(feats_np.shape[1])]
    feat_df   = pd.DataFrame(feats_np, columns=col_names)
    feat_df.insert(0, 'nii.gz_name', all_names)

    feat_df = feat_df.merge(df[['nii.gz_name', 'label']], on='nii.gz_name', how='left')
    feat_df.to_csv(SAVE_FEAT, index=False)
    print(f'\nFeatures saved to {SAVE_FEAT}  shape: {feats_np.shape}')


if __name__ == '__main__':
    extract()
