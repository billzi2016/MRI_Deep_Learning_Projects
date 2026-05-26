# ABIDE I 数据下载说明

**Autism Brain Imaging Data Exchange — Phase I**  
数据平台：NITRC / AWS S3（无需审核，注册即可下载）

---

## 申请步骤

1. 访问 http://fcon_1000.projects.nitrc.org/indi/abide 了解数据集
2. 在 NITRC 注册账号：https://www.nitrc.org/account/register.php
3. 同意使用条款后即可下载，**无需等待审核**

---

## 下载原始数据（AWS S3）

```bash
# 安装 AWS CLI
pip install awscli

# 下载全部 ABIDE I 原始 T1w 数据（匿名访问，无需 AWS 账号）
aws s3 sync \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/RawData/ \
    ./ABIDE_I_raw/ \
    --no-sign-request \
    --exclude "*" \
    --include "*/anat/*T1w*"
```

---

## 下载预处理版本（推荐）

ABIDE Preprocessed 提供已完成颅骨剥除 + 配准的版本：

```bash
# 下载 cpac 流程 / filt_global 版本
aws s3 sync \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/ \
    ./ABIDE_I/ \
    --no-sign-request
```

> 结构像预处理版本路径：  
> `s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/`

---

## 获取 CSV 标签文件

表型数据（诊断标签、站点信息）在官网直接下载：

```
http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.tar.gz
```

解压后 `Phenotypic_V1_0b_preprocessed1.csv` 包含：
- `FILE_ID`：对应影像文件名
- `DX_GROUP`：1 = ASD，2 = 正常对照
- `SITE_ID`：采集站点

---

## 目录结构

```
ABIDE_I/
├── README.md
├── Caltech_0051456_func_preproc.nii.gz
├── CMU_a_0050642_func_preproc.nii.gz
└── ...
```
