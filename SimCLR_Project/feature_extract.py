import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import zoom

# ── Config ──────────────────────────────────────────────────────────────
DATASET_CONFIGS = [
    {'name': 'ABCD',     'csv': '../ABCD_Project/data.csv',     'nii_dir': '../ABCD'},
    {'name': 'ABIDE_I',  'csv': '../ABIDE_I_Project/data.csv',  'nii_dir': '../ABIDE_I'},
    {'name': 'ABIDE_II', 'csv': '../ABIDE_II_Project/data.csv', 'nii_dir': '../ABIDE_II'},
    {'name': 'HCP_Y',    'csv': '../HCP_Y_Project/data.csv',    'nii_dir': '../HCP_Y'},
    {'name': 'ADNI',     'csv': '../ADNI_Project/data.csv',     'nii_dir': '../ADNI'},
]

MODEL_PATH   = 'simclr_encoder.pt'
OUTPUT_DIR   = 'features'
TARGET_SIZE  = (96, 96, 96)
BATCH_SIZE   = 8
GPU_IDS      = list(range(8))
NUM_WORKERS  = 16
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


class Encoder3D(nn.Module):
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
        self.fc      = nn.Sequential(
            nn.Linear(512 * 8, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),    nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x    # (B, 4096)


def extract_dataset(model, df, nii_dir, dataset_name, device, gpu_ids):
    loader = DataLoader(
        MRIDataset(df, nii_dir),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    all_feats = []
    all_names = []

    with torch.no_grad():
        for imgs, names in loader:
            imgs  = imgs.to(device)
            feats = (
                model.module(imgs)
                if isinstance(model, nn.DataParallel)
                else model(imgs)
            )
            all_feats.append(feats.cpu().numpy())
            all_names.extend(names)

    feats_np  = np.concatenate(all_feats, axis=0)              # (N, 4096)
    col_names = [f'feat_{i}' for i in range(feats_np.shape[1])]
    feat_df   = pd.DataFrame(feats_np, columns=col_names)
    feat_df.insert(0, 'nii.gz_name', all_names)
    feat_df.insert(1, 'dataset', dataset_name)

    # carry over any label columns that exist in the source CSV
    for col in ['label', 'stage', 'score']:
        if col in df.columns:
            feat_df = feat_df.merge(df[['nii.gz_name', col]], on='nii.gz_name', how='left')

    out_path = os.path.join(OUTPUT_DIR, f'features_{dataset_name}.csv')
    feat_df.to_csv(out_path, index=False)
    print(f'  {dataset_name}: {feats_np.shape[0]} samples → {out_path}')
    return feat_df


def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available = min(len(GPU_IDS), torch.cuda.device_count())
    gpu_ids   = GPU_IDS[:available]

    model = Encoder3D()
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    model.eval()

    all_dfs = []

    for cfg in DATASET_CONFIGS:
        if not os.path.exists(cfg['csv']) or not os.path.isdir(cfg['nii_dir']):
            print(f"[skip] {cfg['name']} — csv or nii_dir not found")
            continue

        df    = pd.read_csv(cfg['csv'])
        print(f"\n=== {cfg['name']} ({len(df)} samples) ===")
        fdf   = extract_dataset(model, df, cfg['nii_dir'], cfg['name'], device, gpu_ids)
        all_dfs.append(fdf)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_all  = os.path.join(OUTPUT_DIR, 'features_all.csv')
        combined.to_csv(out_all, index=False)
        print(f'\nCombined feature file: {out_all}  '
              f'({len(combined)} samples × {combined.shape[1]} columns)')


if __name__ == '__main__':
    extract()
