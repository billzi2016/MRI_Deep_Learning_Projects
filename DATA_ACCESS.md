# 数据集获取说明

本项目所用五个数据集均为受管控的脑影像公开数据集，各有独立的申请流程和使用协议。以下为各数据集的官方获取渠道及所需资质。

---

## ABCD — Adolescent Brain Cognitive Development

| 项目 | 内容 |
|------|------|
| 官网 | https://abcdstudy.org |
| 数据平台 | NIMH Data Archive（NDA）https://nda.nih.gov |
| 数据集页面 | https://nda.nih.gov/abcd |

**申请要求：**
- 机构隶属（高校 / 科研机构），需 PI（负责人）具有正式职位
- 在 NDA 平台注册账号，由 PI 提交 **Data Use Agreement（DUA）**
- 需填写研究用途说明，经 NDA 审核后获批（通常 2–4 周）
- 所有数据操作须通过 NDA 平台的安全数据传输工具（`downloadcmd`）进行

**使用限制：**
- 仅限批准的研究用途，禁止商用
- 不得尝试重新识别受试者身份
- 发表论文须致谢 ABCD Study 及资助来源（NIMH + 多个 NIH 分支）

---

## ABIDE I & II — Autism Brain Imaging Data Exchange

| 项目 | 内容 |
|------|------|
| 官网 | http://fcon_1000.projects.nitrc.org/indi/abide |
| 数据平台 | NITRC / AWS S3 公开存储 |
| 预处理版本 | http://preprocessed-connectomes-project.org/abide |

**申请要求：**
- **无需正式申请**，数据公开可下载
- 下载前需在 NITRC 注册账号并同意使用条款
- 预处理版本（ABIDE Preprocessed）可直接通过 AWS S3 匿名下载

**使用限制：**
- 遵守 **CC BY-NC 3.0**（署名，非商业使用）
- 不得尝试重新识别受试者
- 发表论文须引用 ABIDE 原始论文（Di Martino et al., 2014 / 2017）

**下载示例（AWS S3）：**
```bash
aws s3 sync s3://fcp-indi/data/Projects/ABIDE_Initiative/ ./ABIDE_I/ --no-sign-request
aws s3 sync s3://fcp-indi/data/Projects/ABIDE2/           ./ABIDE_II/ --no-sign-request
```

---

## HCP_Y — Human Connectome Project (Young Adult)

| 项目 | 内容 |
|------|------|
| 官网 | https://www.humanconnectome.org |
| 数据平台 | ConnectomeDB https://db.humanconnectome.org |

**申请要求（Open Access）：**
- 在 ConnectomeDB 注册账号
- 同意 **WU-Minn HCP Open Access Data Use Terms**
- 审核通常即时生效，注册后可立即下载

**申请要求（Restricted Access，含基因 / 详细行为数据）：**
- 额外提交 **Restricted Access Data Use Terms**
- 需机构 IRB 批准文件
- 人工审核，通常 1–2 周

**使用限制：**
- Open Access 数据可用于学术研究，发表须致谢 WU-Minn HCP 及 NIH
- Restricted Access 数据禁止共享给未申请的第三方
- 不得尝试重新识别受试者

**下载工具：** ConnectomeDB 提供 Aspera 高速下载或 Amazon S3 下载链接

---

## ADNI — Alzheimer's Disease Neuroimaging Initiative

| 项目 | 内容 |
|------|------|
| 官网 | https://adni.loni.usc.edu |
| 数据平台 | LONI Image and Data Archive（IDA）https://ida.loni.usc.edu |
| 申请入口 | https://adni.loni.usc.edu/data-samples/access-data |

**申请要求：**
- 在 LONI IDA 注册账号
- 提交 **ADNI Data Use Agreement**，需包含：
  - 研究人员信息（姓名、机构、职位）
  - 研究目的说明
  - PI 签名及机构盖章
- 审核周期通常 **1–3 周**
- 申请获批后方可通过 LONI 平台检索和下载影像及临床数据

**使用限制：**
- 仅限非商业学术研究
- 数据不得转让给未经批准的第三方
- 发表论文须致谢 ADNI 资助来源（NIA / NIH / 私人基金会联合资助），并引用 ADNI 数据集

---

## 汇总对比

| 数据集 | 需要申请 | 审核周期 | 商业使用 | 数据协议 |
|--------|----------|----------|----------|----------|
| ABCD | 是（NDA DUA） | 2–4 周 | 否 | NDA DUA |
| ABIDE I/II | 否（注册即可） | 即时 | 否 | CC BY-NC 3.0 |
| HCP_Y（Open） | 是（在线同意） | 即时 | 否 | WU-Minn Open Access |
| HCP_Y（Restricted） | 是（人工审核） | 1–2 周 | 否 | WU-Minn Restricted |
| ADNI | 是（书面 DUA） | 1–3 周 | 否 | ADNI DUA |

---

## 注意事项

- 所有数据集均**禁止商业使用**，且明确禁止重新识别受试者身份
- 本项目代码库中**不包含任何原始数据**，数据目录（`ABCD/`、`ABIDE_I/` 等）仅为占位符
- 使用前请确认所在机构的 IRB / 伦理审批状态
- 各数据集的引用要求详见各自官网的 **Publication Policy**
