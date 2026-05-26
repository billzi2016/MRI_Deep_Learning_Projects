# MRI 深度学习项目总览

本仓库包含基于多个大型脑影像公开数据集的深度学习分析项目，涵盖**监督分类**、**回归预测**、**自监督预训练**及**可解释性分析**。

---

## 数据集说明

| 文件夹 | 数据集 | 内容简介 |
|--------|--------|----------|
| `ABCD/` | Adolescent Brain Cognitive Development | 青少年脑发育队列，含自杀/非自杀标签及相关量表分 |
| `ABIDE_I/` | Autism Brain Imaging Data Exchange I | 自闭症谱系障碍 vs 正常对照，多站点采集 |
| `ABIDE_II/` | Autism Brain Imaging Data Exchange II | ABIDE 第二期，扩大样本量，改进采集协议 |
| `HCP_Y/` | Human Connectome Project — Young Adult | 健康青年高分辨率结构 + 功能像，HCP 最小预处理流程 |
| `ADNI/` | Alzheimer's Disease Neuroimaging Initiative | 阿尔茨海默病纵向队列，含 CN / SMC / EMCI / LMCI / AD 五类 |

> 所有数据均为预处理后的 `.nii.gz` 格式（颅骨剥除 + 偏场校正 + 配准至 MNI152 标准空间）。

---

## 项目结构

```
MRI_Deep_Learning_Projects/
├── ABCD_Project/          # ABCD 监督学习
├── ADNI_Project/          # ADNI 多分类 + 回归 + SHAP 可视化
├── SimCLR_Project/        # 跨数据集自监督预训练 + 特征提取
├── ABCD/                  # 数据目录（.nii.gz）
├── ABIDE_I/               # 数据目录
├── ABIDE_II/              # 数据目录
├── HCP_Y/                 # 数据目录
└── ADNI/                  # 数据目录
```

---

## ABCD_Project — 自杀风险分类与评分回归

基于 ABCD 数据集，以结构 MRI 为输入，训练 **3D VGG16** 模型完成两类任务：

**模型架构**：3D VGG16（5个卷积块 + AdaptiveAvgPool3d → 4096-dim → 分类/回归头）

| 文件 | 任务 | 标签 | 损失函数 |
|------|------|------|----------|
| `train_onehot.py` | 二分类：自杀 vs 非自杀 | label（0/1） | CrossEntropyLoss |
| `feature_onehot.py` | 提取全量数据特征 | — | — |
| `train_float.py` | 回归：风险评分预测 | score（MinMax→[0,1]） | MSELoss |
| `feature_float.py` | 提取全量数据特征 | — | — |

- 分类模型输出：`best_model_onehot.pt`
- 回归模型输出：`best_model_float.pt`
- 特征维度：**4096-dim**，覆盖全部样本，保存为 CSV

---

## ADNI_Project — 阿尔茨海默病分期分类与 SHAP 可解释性

基于 ADNI 数据集，完成五类认知分期的**分类**与**量表评分回归**，并通过 SHAP 定位模型关注的脑区。

**五类标签定义**（详见 `ADNI_Project/README.md`）：

| 标签 | 含义 | 核心标准 |
|------|------|----------|
| CN | 认知正常 | CDR=0，MMSE 24-30，WLM 正常 |
| SMC | 主观记忆担忧 | CDR=0，客观测试正常，自我报告记忆下降 |
| EMCI | 早期轻度认知障碍 | CDR=0.5，WLM 轻度受损 |
| LMCI | 晚期轻度认知障碍 | CDR=0.5，WLM 显著受损 |
| AD | 阿尔茨海默病（轻度） | CDR=0.5-1.0，MMSE 20-26 |

| 文件 | 任务 |
|------|------|
| `train_onehot.py` | 五分类（类别加权 CrossEntropyLoss） |
| `train_float.py` | 量表评分回归（MSELoss） |
| `shap_vis_onehot.py` | 分类模型 SHAP 脑区可视化（三视图 + 平均热力图 NIfTI） |
| `shap_vis_float.py` | 回归模型 SHAP 脑区可视化 |

- SHAP 方法：`GradientExplainer`，输出逐被试 PNG 切片及组平均 `.nii.gz`

---

## SimCLR_Project — 跨数据集自监督预训练

以**全部五个数据集**的结构 MRI 为输入，使用 **SimCLR** 框架进行自监督对比学习预训练，获取通用脑影像表征，再为下游任务提取特征。

**训练流程：**

```
原始 MRI → 随机增强（×2视图）→ 3D VGG16 Encoder → Projector（4096→512→128）→ NT-Xent Loss
```

**数据增强策略（3D）：**
- 随机翻转（X / Y / Z 三轴独立）
- 随机高斯噪声
- 随机强度缩放与偏移
- 随机小角度三维旋转（±10°）

**NT-Xent Loss（归一化温度缩放交叉熵）：**
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}, \quad \tau = 0.5$$

| 文件 | 功能 |
|------|------|
| `train_simclr.py` | 自监督预训练，保存 `simclr_encoder.pt` |
| `feature_extract.py` | 加载 encoder，为五个数据集分别提取 4096-dim 特征，合并输出 `features_all.csv` |

---

## 公共配置

| 参数 | 值 |
|------|----|
| 输入尺寸 | 96 × 96 × 96（双线性插值缩放） |
| Epoch 上限 | 1,000,000 |
| Early Stopping patience | 20 |
| GPU | 0–7（8× H100，DataParallel） |
| 混合精度 | AMP（autocast + GradScaler） |
| 优化器 | AdamW（weight decay=1e-4） |

---

## 依赖

```
torch >= 2.0
nibabel
pandas
numpy
scipy
scikit-learn
shap
matplotlib
```
