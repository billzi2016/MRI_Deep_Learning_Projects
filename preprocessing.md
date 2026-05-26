# 结构 MRI 预处理流程

所有数据集（ABCD / ABIDE_I / ABIDE_II / HCP_Y / ADNI）统一使用以下流程处理，工具链为 **ANTs + FSL**，配准模板为 **MNI152 1mm**。

---

## 依赖

| 工具 | 版本 | 用途 |
|------|------|------|
| ANTs | ≥ 2.4 | 偏场校正、颅骨剥除、非线性配准 |
| FSL | ≥ 6.0 | 线性预配准（FLIRT） |

```bash
# 验证安装
antsRegistration --version
flirt -version
```

---

## 流程总览

```
原始 T1w .nii.gz
        │
        ▼
① N4 偏场校正（ANTs N4BiasFieldCorrection）
        │
        ▼
② 颅骨剥除（ANTs antsBrainExtraction）
        │
        ▼
③ 线性预配准到 MNI152（FSL FLIRT，12-DOF）
        │
        ▼
④ 非线性配准到 MNI152（ANTs SyN）
        │
        ▼
⑤ 强度归一化（0–1 Min-Max）
        │
        ▼
输出：96×96×96 .nii.gz，送入模型
```

---

## 步骤一：N4 偏场校正

消除 MRI 扫描的低频强度不均匀性（射频偏场），是所有后续步骤的前提。

```bash
N4BiasFieldCorrection \
    -d 3 \
    -i sub-001_T1w.nii.gz \
    -o sub-001_T1w_N4.nii.gz \
    -s 4 \
    -b [180] \
    -c [50x50x50x50, 0.0]
```

| 参数 | 含义 |
|------|------|
| `-d 3` | 三维图像 |
| `-s 4` | 下采样因子，加速计算 |
| `-b [180]` | B-spline 网格间距（mm） |
| `-c [50x50x50x50, 0.0]` | 四级迭代，收敛阈值 0 |

---

## 步骤二：颅骨剥除

使用 ANTs 概率图集模板进行脑提取，精度优于 FSL BET。

```bash
antsBrainExtraction.sh \
    -d 3 \
    -a sub-001_T1w_N4.nii.gz \
    -e MNI152_T1_1mm.nii.gz \
    -m MNI152_T1_1mm_brain_mask.nii.gz \
    -o sub-001_
```

输出：
- `sub-001_BrainExtractionBrain.nii.gz`（去颅骨脑像）
- `sub-001_BrainExtractionMask.nii.gz`（二值脑掩膜）

> 所用模板：MNI152 标准脑 + 对应脑掩膜，来自 FSL 安装目录  
> `$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz`

---

## 步骤三：线性预配准（FSL FLIRT，12-DOF）

在非线性配准之前做仿射预配准，加速后续 SyN 收敛。

```bash
flirt \
    -in  sub-001_BrainExtractionBrain.nii.gz \
    -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
    -out sub-001_affine.nii.gz \
    -omat sub-001_affine.mat \
    -dof 12 \
    -cost corratio \
    -interp trilinear
```

---

## 步骤四：非线性配准到 MNI152（ANTs SyN）

SyN（Symmetric Normalization）是结构 MRI 配准的行业标准，保证拓扑一致性。

```bash
antsRegistrationSyNQuick.sh \
    -d 3 \
    -f $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
    -m sub-001_BrainExtractionBrain.nii.gz \
    -o sub-001_MNI_ \
    -t s \
    -n 8
```

| 参数 | 含义 |
|------|------|
| `-f` | 固定图像（MNI152 参考脑） |
| `-m` | 移动图像（被试脑像） |
| `-t s` | Rigid + Affine + SyN 三阶段配准 |
| `-n 8` | 使用 8 线程 |

输出：
- `sub-001_MNI_Warped.nii.gz`（配准到 MNI 空间的脑像）
- `sub-001_MNI_1Warp.nii.gz` / `sub-001_MNI_0GenericAffine.mat`（变换场，备用）

---

## 步骤五：强度归一化

将配准后的图像像素值线性缩放到 \[0, 1\]，在 Python 数据加载时完成，无需单独保存：

```python
img = nib.load('sub-001_MNI_Warped.nii.gz').get_fdata(dtype=np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
```

---

## 批量处理脚本

```bash
#!/bin/bash
# preprocess_batch.sh
# 用法：bash preprocess_batch.sh /path/to/raw_nii /path/to/output

RAW_DIR=$1
OUT_DIR=$2
MNI_BRAIN=$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz
MNI_FULL=$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz
MNI_MASK=$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz

mkdir -p "$OUT_DIR"

for nii in "$RAW_DIR"/*.nii.gz; do
    sub=$(basename "$nii" .nii.gz)
    echo "=== Processing $sub ==="
    tmp="$OUT_DIR/${sub}_tmp"
    mkdir -p "$tmp"

    # ① N4
    N4BiasFieldCorrection -d 3 \
        -i  "$nii" \
        -o  "$tmp/${sub}_N4.nii.gz" \
        -s 4 -b [180] -c [50x50x50x50,0.0]

    # ② 颅骨剥除
    antsBrainExtraction.sh -d 3 \
        -a "$tmp/${sub}_N4.nii.gz" \
        -e "$MNI_FULL" \
        -m "$MNI_MASK" \
        -o "$tmp/${sub}_"

    # ③ 线性预配准
    flirt -in  "$tmp/${sub}_BrainExtractionBrain.nii.gz" \
          -ref "$MNI_BRAIN" \
          -out "$tmp/${sub}_affine.nii.gz" \
          -omat "$tmp/${sub}_affine.mat" \
          -dof 12 -cost corratio -interp trilinear

    # ④ SyN 非线性配准
    antsRegistrationSyNQuick.sh -d 3 \
        -f "$MNI_BRAIN" \
        -m "$tmp/${sub}_BrainExtractionBrain.nii.gz" \
        -o "$tmp/${sub}_MNI_" \
        -t s -n 8

    # 复制最终结果
    cp "$tmp/${sub}_MNI_Warped.nii.gz" "$OUT_DIR/${sub}.nii.gz"
    rm -rf "$tmp"
    echo "  -> $OUT_DIR/${sub}.nii.gz"
done

echo "=== 全部完成 ==="
```

运行方式：

```bash
bash preprocess_batch.sh /data/ADNI_raw /data/ADNI_processed
```

---

## 质控（QC）

预处理完成后，对以下指标进行抽检：

1. **配准质量**：用 FSLeyes 叠加被试像与 MNI 模板，检查脑边界对齐
2. **颅骨剥除**：确认无残留颅骨、无脑组织误删
3. **体素尺寸**：所有输出均为 `1mm × 1mm × 1mm`（各数据集统一）
4. **视觉一致性**：随机抽取 10 例，确认三视图（轴位/冠状/矢状）无明显异常

---

## 各数据集说明

| 数据集 | 是否已预处理 | 说明 |
|--------|-------------|------|
| HCP_Y | 是 | HCP 官方提供 MNI 配准版本，可直接使用 |
| ABCD | 否 | 需跑完整流程 |
| ABIDE_I/II | 否 | 建议使用 CPAC 或本流程统一处理，保证多站点一致性 |
| ADNI | 否 | 官网提供部分预处理版，建议统一重新跑以保证一致性 |
