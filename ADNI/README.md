# ADNI 数据下载说明

**Alzheimer's Disease Neuroimaging Initiative**  
数据平台：LONI Image and Data Archive（IDA）

---

## 申请步骤

1. 访问 https://adni.loni.usc.edu/data-samples/access-data
2. 点击 **"Apply for Access"**，在 LONI IDA 注册账号
3. 下载并填写 **ADNI Data Use Agreement（DUA）**，需包含：
   - 研究人员姓名、机构、职位
   - PI 签名
   - 机构授权盖章
4. 将签署完成的 DUA 发送至 adni-help@loni.usc.edu
5. 审核通过后（约 1–3 周），账号激活即可登录 LONI IDA 下载数据

---

## 登录与检索

1. 登录 https://ida.loni.usc.edu
2. 进入 **Search → Advanced Image Search**
3. 按以下条件过滤：
   - **Project**：ADNI
   - **Modality**：MRI
   - **Weighting**：T1
   - **Preprocessing**：选择 "Scaled" 或 "Scaled_2" 版本（已做梯度扭曲校正）
4. 选中目标被试，加入下载列表（Collections）

---

## 下载工具

```bash
# 安装 LONI 下载工具
pip install loni-dl

# 或使用 LONI IDA 网页端直接打包下载（适合小批量）
```

批量下载推荐使用 LONI 官方 Java 下载工具 **LONI Download Manager**：

```bash
java -jar loni-download-manager.jar \
    --username <your_email> \
    --password <your_password> \
    --collection <collection_id> \
    --directory ./ADNI/
```

---

## 获取 CSV 标签文件

在 LONI IDA 下载以下数据表（路径：**Download → Study Data**）：

| 表名 | 内容 |
|------|------|
| `ADNIMERGE.csv` | 每次随访的诊断、MMSE、CDR、ADAS 分等核心指标 |
| `DXSUM_PDXCONV_ADNIALL.csv` | 各访问时间点的诊断标签（CN/MCI/AD） |
| `UCSFFSX_11_02_15.csv` | FreeSurfer 分区体积（皮层厚度、海马体积等） |

以 `PTID`（被试 ID）和 `IMAGEUID`（影像 ID）为主键，将诊断标签与影像文件名对应，生成 `data.csv`（列：`nii.gz_name`，`stage`，`score`）。

**stage 映射关系：**

| ADNIMERGE 中的 DX | 本项目 stage 标签 |
|-------------------|-----------------|
| CN | CN |
| SMC | SMC |
| EMCI | EMCI |
| LMCI | LMCI |
| MCI（ADNI-1）| LMCI |
| Dementia | AD |

---

## 推荐下载的影像类型

| 类型 | 说明 |
|------|------|
| MP-RAGE / IR-SPGR | 主力 T1w 序列，各站点标准采集 |
| Scaled | 已做梯度扭曲校正，推荐优先选择 |
| Accelerated | ADNI-3 新增高分辨率序列 |

---

## 目录结构

```
ADNI/
├── README.md
├── 002_S_0413_2011-03-21.nii.gz
├── 002_S_0816_2012-08-06.nii.gz
└── ...
```

> 文件命名规则：`<PTID>_<ScanDate>.nii.gz`，与 `data.csv` 中 `nii.gz_name` 列对应。
