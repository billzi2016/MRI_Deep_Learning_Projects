import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH        = 'data.csv'
NII_DIR         = '../ADNI'
MODEL_PATH      = 'best_model_adni_float.pt'
OUTPUT_DIR      = 'shap_output_float'
TARGET_SIZE     = (96, 96, 96)
GPU_IDS         = list(range(8))
N_BACKGROUND    = 30
N_EXPLAIN       = 50          # total subjects to explain (regression: no per-class cap)
ALPHA           = 0.5
# ────────────────────────────────────────────────────────────────────────


class ADNIDataset(Dataset):
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
        img      = self._normalize_img(img)
        img      = torch.from_numpy(img).unsqueeze(0)
        score    = float(row['score_norm'])
        return img, score, row['nii.gz_name']

    def _resize(self, img):
        factors = [t / s for t, s in zip(self.target_size, img.shape[:3])]
        return zoom(img, factors, order=1)

    def _normalize_img(self, img):
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.head(x).squeeze(1)


def _save_slice_figure(img_vol, shap_vol, subject_name, true_score, pred_score, out_dir):
    D, H, W = img_vol.shape
    slices = {
        'axial':    (img_vol[D // 2, :, :],  shap_vol[D // 2, :, :]),
        'coronal':  (img_vol[:, H // 2, :],  shap_vol[:, H // 2, :]),
        'sagittal': (img_vol[:, :, W // 2],  shap_vol[:, :, W // 2]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f'{subject_name}  |  True score: {true_score:.4f}  Pred score: {pred_score:.4f}',
        fontsize=10
    )

    for ax, (view_name, (img_sl, shap_sl)) in zip(axes, slices.items()):
        ax.imshow(img_sl.T, cmap='gray', origin='lower', interpolation='nearest')
        shap_abs  = np.abs(shap_sl)
        shap_norm = shap_abs / (shap_abs.max() + 1e-8)
        im = ax.imshow(
            shap_norm.T, cmap='hot', alpha=ALPHA,
            origin='lower', interpolation='nearest',
            vmin=0, vmax=1
        )
        ax.set_title(view_name)
        ax.axis('off')

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label='|SHAP| (norm)')
    plt.tight_layout()

    safe_name = subject_name.replace('.nii.gz', '')
    fname     = os.path.join(out_dir, f'{safe_name}_true{true_score:.3f}_pred{pred_score:.3f}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _save_mean_shap_figure(mean_shap_vol, out_dir):
    D, H, W = mean_shap_vol.shape
    slices = {
        'axial':    mean_shap_vol[D // 2, :, :],
        'coronal':  mean_shap_vol[:, H // 2, :],
        'sagittal': mean_shap_vol[:, :, W // 2],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Mean |SHAP| — score regression', fontsize=12)

    for ax, (view_name, sl) in zip(axes, slices.items()):
        sl_norm = sl / (sl.max() + 1e-8)
        im = ax.imshow(sl_norm.T, cmap='hot', origin='lower', interpolation='nearest')
        ax.set_title(view_name)
        ax.axis('off')

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label='Mean |SHAP| (norm)')
    plt.tight_layout()

    fname = os.path.join(out_dir, 'mean_shap_score.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def run_shap():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df['stage'] = df['stage'].str.strip().str.upper().replace({'MCI': 'LMCI'})

    scaler           = MinMaxScaler()
    df['score_norm'] = scaler.fit_transform(df[['score']]).squeeze()

    dataset = ADNIDataset(df, NII_DIR)

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available = min(len(GPU_IDS), torch.cuda.device_count())
    gpu_ids   = GPU_IDS[:available]

    model = VGG16_3D()
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    model.eval()

    # ── Background batch ──
    bg_indices = np.random.choice(len(dataset), size=min(N_BACKGROUND, len(dataset)), replace=False)
    bg_imgs    = torch.stack([dataset[i][0] for i in bg_indices]).to(device)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    explainer  = shap.GradientExplainer(base_model, bg_imgs)
    print(f'GradientExplainer ready  (background: {len(bg_indices)} samples)')

    mean_shap = np.zeros(TARGET_SIZE, dtype=np.float64)
    n_done    = 0

    for idx in range(len(dataset)):
        if n_done >= N_EXPLAIN:
            break

        img, true_score, name = dataset[idx]
        img_t = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_score = base_model(img_t).item()

        # GradientExplainer returns a list of length 1 for single-output regression
        shap_values = explainer.shap_values(img_t)
        sv = shap_values[0][0, 0] if isinstance(shap_values, list) else shap_values[0, 0]

        mean_shap += np.abs(sv)
        n_done    += 1

        fname = _save_slice_figure(
            img.squeeze(0).cpu().numpy(), sv,
            name, true_score, pred_score, OUTPUT_DIR
        )
        print(f'  [{n_done}/{N_EXPLAIN}] {name}  -> {fname}')

    # ── Mean SHAP map ──
    if n_done > 0:
        mean_vol = mean_shap / n_done
        fname    = _save_mean_shap_figure(mean_vol, OUTPUT_DIR)
        print(f'Mean SHAP saved: {fname}')

        nii_out = nib.Nifti1Image(mean_vol.astype(np.float32), affine=np.eye(4))
        nib.save(nii_out, os.path.join(OUTPUT_DIR, 'mean_shap_score.nii.gz'))

    print(f'\nAll SHAP outputs saved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    run_shap()
