## 🚀 AdaRode_XL Overview

**AdaRode_XL** is the enhanced model suite, including:
- 📥 Loading of existing (pretrained) models
- ➕ Incremental training programs for continued/online fine-tuning

---

## 📁 Folder Guide (each folder = one workflow)

A high-level view (names may vary slightly by your repo):

AdaRode_XL/
├─ Data/
│ ├─ AdaPIK/ # ✅ Merged dataset (with train/val/test splits)
│ │ ├─ train.csv
│ │ ├─ val.json
│ │ └─ test.csv
│ └─ ... # other data sources
    ├─ MIV/
    │ ├─ test_set.csv # ⚠️ Adversarially mutated test data
    ├─ configs/ # YAML/JSON configs for models & data (optional)
└─ Test.py # unified test entry


**Augment_AdaRodeXL.py - ➕ Incremental training programs for continued/online fine-tuning**

---------------------------------------------------
## 🔧 Setup Base Weights & Adversarial Training Data

At the beginning, please set the **base model weights** and the **data source for incremental adversarial training** 🧠⚔️.


12: data_path = "/root/autodl-fs/AdaRode_XL/Data/AdaRode/AugmentedData.csv"
13: model_path = "/root/autodl-fs/Model/XLnet.pth"

---------------------------------------------------




- **`Data/AdaPIK`** 🧾  
  The **merged dataset** after preprocessing. It **already includes**:
  - `train` (training split)
  - `dev` / `val` (validation split)
  - `test` (standard test split)

- **`MIV/test_set`** 🧪  
  Stores **adversarially mutated** test data. Use this to evaluate robustness under mutation/perturbation scenarios.

---

## ▶️ Run Tests on Different Data

Use **`Test.py`** to run evaluations. By **changing the data source path**, you can obtain results under different datasets (standard vs adversarial, or different corpora).
