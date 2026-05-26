import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import zoom, rotate

# ── Config ──────────────────────────────────────────────────────────────
DATASET_CONFIGS = [
    {'name': 'ABCD',     'csv': '../ABCD_Project/data.csv',     'nii_dir': '../ABCD'},
    {'name': 'ABIDE_I',  'csv': '../ABIDE_I_Project/data.csv',  'nii_dir': '../ABIDE_I'},
    {'name': 'ABIDE_II', 'csv': '../ABIDE_II_Project/data.csv', 'nii_dir': '../ABIDE_II'},
    {'name': 'HCP_Y',    'csv': '../HCP_Y_Project/data.csv',    'nii_dir': '../HCP_Y'},
    {'name': 'ADNI',     'csv': '../ADNI_Project/data.csv',     'nii_dir': '../ADNI'},
]

SAVE_PATH    = 'simclr_encoder.pt'
TARGET_SIZE  = (96, 96, 96)
BATCH_SIZE   = 32           # per-GPU; effective = 32 × 8 = 256
NUM_EPOCHS   = 1_000_000
LR           = 3e-4
TEMPERATURE  = 0.5
PROJ_DIM     = 128
PATIENCE     = 20
GPU_IDS      = list(range(8))
NUM_WORKERS  = 16
# ────────────────────────────────────────────────────────────────────────


# ── 3-D MRI augmentation ─────────────────────────────────────────────────
class Augment3D:
    """Returns two independently augmented views of a (1,D,H,W) tensor."""

    def __call__(self, img: torch.Tensor):
        return self._augment(img), self._augment(img)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clone()

        # random flip along each spatial axis
        for dim in [1, 2, 3]:
            if random.random() > 0.5:
                img = torch.flip(img, [dim])

        # random Gaussian noise
        if random.random() > 0.5:
            img = (img + torch.randn_like(img) * random.uniform(0.01, 0.05)).clamp(0, 1)

        # random intensity scale
        if random.random() > 0.5:
            img = (img * random.uniform(0.8, 1.2)).clamp(0, 1)

        # random intensity shift
        if random.random() > 0.5:
            img = (img + random.uniform(-0.1, 0.1)).clamp(0, 1)

        # random 3-D rotation (small angle, scipy operates on numpy)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            axes  = random.choice([(0, 1), (0, 2), (1, 2)])
            vol   = img.squeeze(0).numpy()
            vol   = rotate(vol, angle, axes=axes, reshape=False, order=1)
            img   = torch.from_numpy(vol).unsqueeze(0)

        return img


# ── Dataset ──────────────────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, csv_path, nii_dir, target_size=TARGET_SIZE):
        self.df          = pd.read_csv(csv_path)
        self.nii_dir     = nii_dir
        self.target_size = target_size
        self.augment     = Augment3D()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name     = self.df.iloc[idx]['nii.gz_name']
        nii_path = os.path.join(self.nii_dir, name)
        img      = nib.load(nii_path).get_fdata(dtype=np.float32)
        img      = self._resize(img)
        img      = self._normalize(img)
        img      = torch.from_numpy(img).unsqueeze(0)   # (1, D, H, W)
        v1, v2   = self.augment(img)
        return v1, v2

    def _resize(self, img):
        factors = [t / s for t, s in zip(self.target_size, img.shape[:3])]
        return zoom(img, factors, order=1)

    def _normalize(self, img):
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        return img


# ── Model ────────────────────────────────────────────────────────────────
class Encoder3D(nn.Module):
    """3-D VGG16 backbone — output is the 4096-dim representation."""
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
        return x                    # (B, 4096)


class SimCLR(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = Encoder3D()
        self.projector = nn.Sequential(
            nn.Linear(4096, 512), nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z                    # (B, proj_dim)


# ── NT-Xent loss ─────────────────────────────────────────────────────────
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE):
    """Normalized temperature-scaled cross-entropy (SimCLR contrastive loss)."""
    B  = z1.size(0)
    z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)    # (2B, D)
    sim = torch.mm(z, z.T) / temperature                    # (2B, 2B)

    # remove self-similarity on diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # positive pair for i → i+B and i+B → i
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ── Training ──────────────────────────────────────────────────────────────
def build_combined_dataset():
    datasets = []
    for cfg in DATASET_CONFIGS:
        if os.path.exists(cfg['csv']) and os.path.isdir(cfg['nii_dir']):
            datasets.append(MRIDataset(cfg['csv'], cfg['nii_dir']))
            print(f"  loaded {cfg['name']}  ({len(datasets[-1])} samples)")
        else:
            print(f"  [skip] {cfg['name']}  — csv or nii_dir not found")
    if not datasets:
        raise RuntimeError('No valid dataset found. Check DATASET_CONFIGS paths.')
    return ConcatDataset(datasets)


def train():
    print('=== Building combined dataset ===')
    combined = build_combined_dataset()
    print(f'Total samples: {len(combined)}\n')

    loader = DataLoader(
        combined,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available = min(len(GPU_IDS), torch.cuda.device_count())
    gpu_ids   = GPU_IDS[:available]

    model = SimCLR(proj_dim=PROJ_DIM)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = GradScaler()

    best_loss      = float('inf')
    patience_count = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            with autocast():
                z1   = model(v1)
                z2   = model(v2)
                loss = nt_xent_loss(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f'Epoch [{epoch:>7d}]  Loss: {avg_loss:.6f}')

        scheduler.step()

        if avg_loss < best_loss:
            best_loss      = avg_loss
            patience_count = 0
            enc_state = (
                model.module.encoder.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.encoder.state_dict()
            )
            torch.save(enc_state, SAVE_PATH)
            print(f'  -> Saved encoder  (loss={avg_loss:.6f})')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch} epochs (patience={PATIENCE})')
                break

    print(f'\nTraining complete. Best loss: {best_loss:.6f}')
    print(f'Encoder saved to: {SAVE_PATH}')


if __name__ == '__main__':
    train()
