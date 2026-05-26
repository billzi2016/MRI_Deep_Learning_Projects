# ABCD 数据下载说明

**Adolescent Brain Cognitive Development Study**  
数据平台：NIMH Data Archive（NDA）

---

## 申请步骤

1. 访问 https://nda.nih.gov，注册账号
2. 由 PI 在 NDA 平台提交 **Data Use Agreement（DUA）**，填写研究目的
3. 审核通过后（约 2–4 周），账号获得 ABCD 数据访问权限

---

## 下载工具安装

```bash
pip install nda-tools
```

---

## 下载影像数据

登录 NDA 后，在 ABCD 数据集页面创建下载包（Data Package），然后：

```bash
# 查看可用的数据包
downloadcmd -dp <package_id> -t

# 下载 T1w 结构像（过滤 MRI 模态）
downloadcmd -dp <package_id> -t imagingcollection01 -d ./
```

> `<package_id>` 在 NDA 网页端创建数据包后获得。

---

## 获取 CSV 标签文件

在 NDA 数据集页面下载以下数据表：

| 表名 | 内容 |
|------|------|
| `abcd_mri01` | 影像文件清单及扫描参数 |
| `abcd_ksad01` | 自杀相关量表（KSADS） |
| `abcd_cbcls01` | 行为量表分 |

将 `src_subject_id` 与影像文件名对应后生成 `data.csv`（列：`nii.gz_name`，`label`，`score`）。

---

## 目录结构

```
ABCD/
├── README.md
├── sub-NDARXXXXXXXX_ses-baselineYear1Arm1_T1w.nii.gz
├── sub-NDARXXXXXXXX_ses-baselineYear1Arm1_T1w.nii.gz
└── ...
```
