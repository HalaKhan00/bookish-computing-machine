# Rare Cardiac Condition ECG Synthesis using GANs

This project focuses on generating synthetic ECG signals for **rare but critical cardiac conditions** using **Generative Adversarial Networks (GANs)**, helping to address the problem of **class imbalance** in diagnostic datasets.

---

## 🫀 What is ECG?

**ECG (Electrocardiogram)** is a recording of the heart's electrical activity through repeated cardiac cycles. It is a key diagnostic tool used to assess cardiac health.

- A standard **12-lead ECG** uses **10 electrodes**: 4 limb leads and 6 chest leads.
  
  **Lead Categories & Viewing Directions:**
  - **I, II, III**: Overall horizontal and diagonal plane activity
  - **aVR, aVL, aVF**: Frontal plane activity (upper body)
  - **V1–V6**: Localized chest-level activity

---

## 📊 Dataset: PTB-XL

We use the **PTB-XL** dataset — a large public ECG database with:

- **21837** samples
- Each sample is a **10-second, 12-lead ECG** signal

---

## 🤖 Approach: ECG Signal Generation with GANs

To enrich the dataset with rare cardiac conditions, we build a GAN architecture tailored for 1D ECG signals.

### ✅ Inception-Style Block (1D)

**Purpose:** Capture features at multiple temporal resolutions.

- Apply parallel Conv1D layers with kernel sizes **1, 3, and 5**
- Use **MaxPooling** for robustness
- Concatenate all outputs → 128 channels

### 🔁 Residual Shortcut Layer

**Purpose:** Improve training stability and gradient flow.

- Adds input directly to output of the Inception block
- Uses 1×1 Conv to align channel dimensions

---

## 🏗️ GAN Architecture

### 🎨 Generator

- Dense → BatchNorm → LeakyReLU → Reshape  
- Output shape: **(1000 × 12)** — representing a full 10-second ECG
- Conv1DTranspose layers to refine temporal structure:
  - 128 → 64 → 12 filters
- Final activation: **`tanh`** (output range [-1, 1])

### 🕵️ Discriminator

- Built to distinguish real vs. fake ECGs  
- Trained with variable learning rates for experimentation

---

## 🔬 Experiment: Comparing Learning Rates

| Approach        | Discriminator LR | Generator LR | Notes |
|----------------|------------------|---------------|-------|
| **Initial**     | 2e-4              | 2e-4          | GAN learned more realistic waveforms |
| **Modified**    | 1e-4              | 4e-4          | Generator created overly smooth, unrealistic waves |

### Key Findings:
- Lower discriminator LR led to better performance.
- Boosted generator LR generated overly smooth, fake-looking ECGs.
- Best AUROC: **0.9899**, Accuracy: **0.948**
- Domain-specific metrics (e.g. **RR-SD**) improved interpretability

---

## 🎯 Goal

To **generate realistic ECG signals** for rare cardiac conditions and support improved diagnostic performance in AI-based systems trained on imbalanced datasets.

---

## 📁 Project Structure
Rare_GAN/
├── arrhythmia_gan_outputs/
├── fake_arrhythmia_candidates/
├── generated_ecgs/
└── Rare_GAN.ipynb
Rare_GAN_Improved/
├── arrhythmia_gan_outputs_tuned/
├── fake_arrhythmia_candidates/
├── generated_ecgs/
└── Rare_GAN.ipynb

---

## 📌 Note

This work is research-focused and intended for augmenting clinical datasets in a safe and controlled manner.




