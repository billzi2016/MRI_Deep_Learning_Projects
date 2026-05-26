# ABIDE II 数据下载说明

**Autism Brain Imaging Data Exchange — Phase II**  
数据平台：NITRC / AWS S3（无需审核，注册即可下载）

---

## 申请步骤

与 ABIDE I 相同：在 NITRC 注册账号并同意使用条款后即可访问，**无需等待审核**。

官网：http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html

---

## 下载原始数据（AWS S3）

```bash
# 下载全部 ABIDE II 原始 T1w 结构像
aws s3 sync \
    s3://fcp-indi/data/Projects/ABIDE2/RawData/ \
    ./ABIDE_II_raw/ \
    --no-sign-request \
    --exclude "*" \
    --include "*/anat/*T1w*"
```

---

## 下载预处理版本（推荐）

```bash
# cpac 流程预处理版本
aws s3 sync \
    s3://fcp-indi/data/Projects/ABIDE2/Outputs/cpac/filt_global/ \
    ./ABIDE_II/ \
    --no-sign-request
```

---

## 获取 CSV 标签文件

表型数据在官网下载：

```
http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.tar.gz
```

解压后 `ABIDEII_Composite_Phenotypic.csv` 包含：
- `FILE_ID`：对应影像文件名
- `DX_GROUP`：1 = ASD，2 = 正常对照
- `SITE_ID`：采集站点（ABIDE II 新增多个站点）
- `AGE_AT_SCAN`：扫描年龄

> **注意**：ABIDE II 比 ABIDE I 新增了更多采集站点，扫描参数差异更大，建议在预处理时严格执行站点校正（ComBat 等方法）。

---

## 目录结构

```
ABIDE_II/
├── README.md
├── BNI_1_0050001_func_preproc.nii.gz
├── EMC_1_0050002_func_preproc.nii.gz
└── ...
```
