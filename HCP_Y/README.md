# HCP_Y 数据下载说明

**Human Connectome Project — Young Adult (1200 Subject Release)**  
数据平台：ConnectomeDB

---

## 申请步骤

1. 访问 https://db.humanconnectome.org 注册账号
2. 登录后在 **WU-Minn HCP Data** 页面点击 **"Open Access Data Use Terms"**
3. 在线同意使用条款，**即时生效**，无需等待审核
4. 如需基因或详细行为数据（Restricted Access），额外提交书面 DUA，人工审核约 1–2 周

---

## 下载工具

ConnectomeDB 支持两种下载方式：

### 方式一：Aspera 高速下载（推荐，速度可达 100MB/s）

```bash
# 安装 Aspera Connect 插件后，在 ConnectomeDB 网页端选择数据包直接下载
# 插件下载：https://www.ibm.com/aspera/connect/
```

### 方式二：Amazon S3

```bash
# 需要在 ConnectomeDB 获取临时 S3 credentials
aws s3 sync \
    s3://hcp-openaccess/HCP_1200/<SubjectID>/T1w/ \
    ./HCP_Y/<SubjectID>/ \
    --region us-east-1
```

---

## 下载 T1w 结构像

在 ConnectomeDB 页面：
1. 选择 **HCP 1200 Subjects** 数据集
2. 过滤 **Structural Preprocessed** 数据包
3. 选择全部或部分被试，加入下载队列

每个被试的预处理 T1w 路径：

```
<SubjectID>/T1w/T1w_acpc_dc_restore_brain.nii.gz   ← 已颅骨剥除，MNI 对齐
<SubjectID>/MNINonLinear/T1w_restore_brain.nii.gz  ← MNI152 非线性配准版本
```

**推荐使用：** `MNINonLinear/T1w_restore_brain.nii.gz`，已完成 HCP 最小预处理流程（FreeSurfer + ANTs + MSM），可直接送入模型，**无需再次预处理**。

---

## 获取 CSV 标签文件

行为及人口学数据在 ConnectomeDB 下载 `HCP_S1200_DataDictionary_April_2018.csv`，关键列：

| 列名 | 内容 |
|------|------|
| `Subject` | 被试 ID |
| `Age` | 年龄段（22-25 / 26-30 / 31-35 / 36+） |
| `Gender` | 性别 |
| `CogTotalComp_Unadj` | 总体认知复合分 |

---

## 目录结构

```
HCP_Y/
├── README.md
├── 100206/
│   └── T1w_restore_brain.nii.gz
├── 100307/
│   └── T1w_restore_brain.nii.gz
└── ...
```

> HCP 数据分辨率为 0.7mm 各向同性，预处理后送入模型前需缩放至 96×96×96（见 `preprocessing.md`）。
