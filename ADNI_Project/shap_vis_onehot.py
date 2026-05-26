import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH        = 'data.csv'
NII_DIR         = '../ADNI'
MODEL_PATH      = 'best_model_adni.pt'
OUTPUT_DIR      = 'shap_output'
TARGET_SIZE     = (96, 96, 96)
BATCH_SIZE      = 1
GPU_IDS         = list(range(8))
NUM_WORKERS     = 8
N_BACKGROUND    = 30        # background samples for GradientExplainer
N_EXPLAIN       = 10        # number of subjects to explain per class
ALPHA           = 0.5       # overlay transparency

LABEL_MAP    = {'CN': 0, 'SMC': 1, 'EMCI': 2, 'LMCI': 3, 'AD': 4}
LABEL_NAMES  = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES  = len(LABEL_MAP)
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
        img      = self._normalize(img)
        img      = torch.from_numpy(img).unsqueeze(0)
        label    = torch.tensor(LABEL_MAP[row['stage']], dtype=torch.long)
        return img, label, row['nii.gz_name']

    def _resize(self, img):
        factors = [t / s for t, s in zip(self.target_size, img.shape[:3])]
        return zoom(img, factors, order=1)

    def _normalize(self, img):
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        return img


class VGG16_3D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.head(x)


def _save_slice_figure(img_vol, shap_vol, subject_name, true_label, pred_label, class_idx, out_dir):
    """Save axial / coronal / sagittal mid-slice overlay for one subject × one class."""
    D, H, W = img_vol.shape
    slices = {
        'axial':    (img_vol[D // 2, :, :],    shap_vol[D // 2, :, :]),
        'coronal':  (img_vol[:, H // 2, :],    shap_vol[:, H // 2, :]),
        'sagittal': (img_vol[:, :, W // 2],    shap_vol[:, :, W // 2]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f'{subject_name}  |  True: {LABEL_NAMES[true_label]}  '
        f'Pred: {LABEL_NAMES[pred_label]}  |  SHAP class: {LABEL_NAMES[class_idx]}',
        fontsize=10
    )

    for ax, (view_name, (img_sl, shap_sl)) in zip(axes, slices.items()):
        ax.imshow(img_sl.T, cmap='gray', origin='lower', interpolation='nearest')
        shap_abs = np.abs(shap_sl)
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

    fname = os.path.join(
        out_dir,
        f'{subject_name}_true{LABEL_NAMES[true_label]}_pred{LABEL_NAMES[pred_label]}'
        f'_shap{LABEL_NAMES[class_idx]}.png'
    )
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _save_mean_shap_figure(mean_shap_vol, class_idx, out_dir):
    """Save mean |SHAP| volume slices averaged across all explained subjects for one class."""
    D, H, W = mean_shap_vol.shape
    slices = {
        'axial':    mean_shap_vol[D // 2, :, :],
        'coronal':  mean_shap_vol[:, H // 2, :],
        'sagittal': mean_shap_vol[:, :, W // 2],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Mean |SHAP| — class: {LABEL_NAMES[class_idx]}', fontsize=12)

    for ax, (view_name, sl) in zip(axes, slices.items()):
        sl_norm = sl / (sl.max() + 1e-8)
        im = ax.imshow(sl_norm.T, cmap='hot', origin='lower', interpolation='nearest')
        ax.set_title(view_name)
        ax.axis('off')

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label='Mean |SHAP| (norm)')
    plt.tight_layout()

    fname = os.path.join(out_dir, f'mean_shap_class{LABEL_NAMES[class_idx]}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def run_shap():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df['stage'] = df['stage'].str.strip().str.upper().replace({'MCI': 'LMCI'})

    dataset = ADNIDataset(df, NII_DIR)

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available = min(len(GPU_IDS), torch.cuda.device_count())
    gpu_ids   = GPU_IDS[:available]

    model = VGG16_3D(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    model.eval()

    # ── Background batch for GradientExplainer ──
    bg_indices = np.random.choice(len(dataset), size=min(N_BACKGROUND, len(dataset)), replace=False)
    bg_imgs    = torch.stack([dataset[i][0] for i in bg_indices]).to(device)
    explainer  = shap.GradientExplainer(
        model.module if isinstance(model, nn.DataParallel) else model,
        bg_imgs
    )
    print(f'GradientExplainer ready  (background: {len(bg_indices)} samples)')

    # ── Per-class mean SHAP accumulator ──
    mean_shap = {c: np.zeros(TARGET_SIZE, dtype=np.float64) for c in range(NUM_CLASSES)}
    mean_cnt  = {c: 0 for c in range(NUM_CLASSES)}

    # Explain up to N_EXPLAIN subjects per class
    explained = {c: 0 for c in range(NUM_CLASSES)}

    for idx in range(len(dataset)):
        img, label, name = dataset[idx]
        label_int = label.item()

        if explained[label_int] >= N_EXPLAIN:
            continue

        img_t = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logit     = (model.module if isinstance(model, nn.DataParallel) else model)(img_t)
            pred_int  = logit.argmax(1).item()

        # shap_values: list of length NUM_CLASSES, each (1, 1, D, H, W)
        shap_values = explainer.shap_values(img_t)

        img_np = img.squeeze(0).cpu().numpy()   # (D, H, W)

        for class_idx in range(NUM_CLASSES):
            sv = shap_values[class_idx][0, 0]   # (D, H, W)
            mean_shap[class_idx] += np.abs(sv)
            mean_cnt[class_idx]  += 1

            # per-subject figure for the predicted class only (keeps output manageable)
            if class_idx == pred_int:
                fname = _save_slice_figure(
                    img_np, sv, name.replace('.nii.gz', ''),
                    label_int, pred_int, class_idx, OUTPUT_DIR
                )
                print(f'  [{idx+1}/{len(dataset)}] {name}  -> {fname}')

        explained[label_int] += 1
        if all(v >= N_EXPLAIN for v in explained.values()):
            break

    # ── Save per-class mean SHAP maps ──
    for class_idx in range(NUM_CLASSES):
        if mean_cnt[class_idx] > 0:
            mean_vol = mean_shap[class_idx] / mean_cnt[class_idx]
            fname    = _save_mean_shap_figure(mean_vol, class_idx, OUTPUT_DIR)
            print(f'Mean SHAP saved: {fname}')

            # also save as NIfTI for further inspection
            nii_out = nib.Nifti1Image(mean_vol.astype(np.float32), affine=np.eye(4))
            nib.save(
                nii_out,
                os.path.join(OUTPUT_DIR, f'mean_shap_{LABEL_NAMES[class_idx]}.nii.gz')
            )

    print(f'\nAll SHAP outputs saved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    run_shap()
