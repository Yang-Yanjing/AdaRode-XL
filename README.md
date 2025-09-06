# 🛡️ AdaRode: Adversarially Robust Detector for Web Injection Attacks

**AdaRode** is an adversarially augmented training framework designed to improve the robustness of Web Application Firewalls (WAFs) against evolving injection attacks such as SQL injection (SQLi) and cross-site scripting (XSS). While conventional WAFs often fail to intercept Malicious Injection Variants (MIVs) that differ from historical attack patterns, AdaRode addresses this challenge from a defender's perspective.

## 🔍 Key Features

- **Mutation Meta-Rule (MMR) Collection**  
  Mutation rules are systematically collected from academic literature, technical blogs, and expert analysis to cover a wide range of real-world SQLi and XSS behaviors.

- **Effective MIV Generation via MCMC Sampling**  
  AdaRode adopts a **Metropolis-Hastings (M-H) sampling algorithm** guided by model-internal confidence outputs to efficiently generate MIVs that are more likely to evade detection. Acceptance rates are dynamically computed to prioritize more adversarial variants.

- **Adaptive Variant Selection**  
  For each traffic sample, the variant causing the maximum change in the model's confidence is selected to probe the model. Variants are accepted or rejected based on adversarial effectiveness.

- **Rollback & Coverage Strategy**  
  The system supports rollback mechanisms when all mutations of a sample are exhausted, ensuring wide coverage across historical traffic.

- **Adversarial Training with MIVs**  
  Successfully generated MIVs are fed back to the training pipeline, augmenting the model to improve generalization against unseen attacks.

- **Modular Pipeline**  
  The full workflow includes:  
  1. MMR collection  
  2. Effective MIV sampling  
  3. Model selection and adversarial training

## 📊 Experimental Highlights

- AdaRode improves MIV detection performance by **over 20%** compared with baseline models.
- Evaluated across multiple attack algorithms and unseen mutation rule settings using 4-fold generalization experiments.
- Incorporates internal confidence outputs as feedback signals, balancing **efficiency** and **diversity** of generated samples.

---

## ✅ Requirements

- Python 3.10 or higher  
- Required packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Set parameters and paths in the config file `Config/adv_config_AdaRode.yaml`. Example:

```yaml
paths:
  model_checkpoint: "/root/autodl-fs/Model/XLnet.pth"

parameters:
  device: "cuda:0"
  max_iterations: 100
  accept_rate: 0.95
  patience: 20

n_workers: 2
```

---

## 🚀 Run Instructions

Run the script from the **root directory**, with the path format `AlgorithmName.py` under `Aug/`. Do **not** run from `Aug/Aug/`.

Example command:

```bash
python Aug/AdaRode.py
```

Replace `AdaRode` with any other algorithm name you want to run.

### 🔁 Automatic Repetition

- Each run performs **10 repeated experiments** automatically.
- Logs will be saved under:

```
Aug/Augmentation/log/
```

---

## 📂 Directory Structure (Key Components)

```
├── Aug/
│   ├── AdaRode.py               # Main entry for the algorithm
│   └── Augmentation/log/        # Logs for each run
├── Config/
│   └── adv_config_AdaRode.yaml  # Configuration file
├── requirements.txt             # Dependency list
```

---

## 📬 Contact

For issues, questions, or collaborations, feel free to open an issue or pull request.
