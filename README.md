# PCOS Diagnosis Using Machine Learning
### DS 4420 — Machine Learning and Data Mining 2
**Northeastern University | Spring 2026**
**Authors:** Gabriela Nirmal, Udita Shah, Praveen Sinha

---

## Project Overview

Polycystic Ovary Syndrome (PCOS) is one of the most common hormonal disorders in women of reproductive age, affecting an estimated 6–10% of this population worldwide. Despite its prevalence, up to 70% of cases go undiagnosed due to the variability of symptoms and overlap with other conditions.

This project applies three machine learning methods to a publicly available PCOS clinical dataset to investigate which approach best classifies patients as PCOS-positive or PCOS-negative. The goal is not only to compare predictive performance across methods, but also to understand what each model reveals about the most important clinical features for diagnosis.

---

## Dataset

**Source:** [Kaggle — PCOS Dataset](https://www.kaggle.com/datasets/shreyasvedpathak/pcos-dataset)

- 541 patients, 41 clinical features
- Binary target: PCOS (Y/N)
- Class distribution: 177 PCOS-positive, 364 PCOS-negative
- Features include hormone levels (LH, FSH, AMH), anthropometric measurements (BMI, Waist:Hip Ratio), symptom indicators (hair growth, skin darkening, weight gain), and follicle counts from ultrasound

---

## Models

This project implements three machine learning models, using a combination of Python and R. At least one model is implemented manually without the use of pre-built modeling packages.

| Model | Description |
|---|---|
| **Multilayer Perceptron (MLP)** | Fully manual implementation using NumPy only — no scikit-learn or TensorFlow. Three hidden layers with ReLU activations, trained with mini-batch gradient descent and momentum. |
| **1D Convolutional Neural Network (CNN)** | Implemented in Keras. Features are reordered to group clinically related variables (follicle counts, hormones, symptoms) together, allowing convolutional filters to detect meaningful local patterns. Class weights and a lowered classification threshold address class imbalance. |
| **Bayesian Logistic Regression** | Implemented in R using the `brms` package. Places prior distributions on model coefficients and estimates the full posterior distribution via MCMC sampling, providing credible intervals for each predictor. |

---

## Repository Structure

```
pcos-analysis/
│
├── data/
│   └── PCOS_data.csv               # Raw dataset (download from Kaggle)
│
├── models/
│   ├── mlp/
│   │   └── pcos_mlp_poc.py         # Manual MLP — NumPy only
│   ├── cnn/
│   │   └── pcos_cnn_keras.ipynb    # 1D CNN — Keras (run on Google Colab)
│   └── bayesian/
│       └── pcos_bayesian.R         # Bayesian logistic regression — brms
│
├── literature_review/
│   └── PCOS_Literature_Review_Phase1.pdf
│
└── README.md
```

---

## Requirements

### Python (MLP + CNN)
```
numpy
pandas
tensorflow / keras
scikit-learn
matplotlib
```

Install with:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

The MLP (`pcos_mlp_poc.py`) uses **only NumPy** — no ML libraries required beyond that.

The CNN notebook is designed to run on **Google Colab**. Upload `PCOS_data.csv` at the start of the session.

### R (Bayesian Model)
```
brms
tidyverse
```

Install with:
```r
install.packages(c("brms", "tidyverse"))
```

Note: `brms` requires a C++ compiler and Stan. First-time setup may take 15–20 minutes.

---

## How to Run

### MLP (Python)
```bash
# place PCOS_data.csv in a folder called data/ next to the script
python pcos_mlp_poc.py
```

### CNN (Google Colab)
1. Open `pcos_cnn_keras.ipynb` in Google Colab
2. Upload `PCOS_data.csv` when prompted
3. Run all cells in order

### Bayesian Model (R)
```r
# open pcos_bayesian.R in RStudio
# ensure PCOS_data.csv is in your working directory
source("pcos_bayesian.R")
```

---

## Results (Phase II — to be updated)

| Model | Accuracy | PCOS Recall | PCOS Precision |
|---|---|---|---|
| MLP (Manual) | TBD | TBD | TBD |
| CNN (Keras) | 83% | 0.78 | 0.74 |
| Bayesian LR | TBD | TBD | TBD |

---

## Project Timeline

| Milestone | Due Date |
|---|---|
| Team Choice Form | February 20, 2026 |
| Phase I — Literature Review + POC Model | March 9, 2026 |
| Project Check-In | Week of March 23, 2026 |
| Final Report + Virtual Poster | April 16, 2026 |

---

## References

1. F. J. Barrera et al., "Application of machine learning and artificial intelligence in the diagnosis and classification of polycystic ovarian syndrome: A systematic review," *Front. Endocrinol.*, vol. 14, p. 1106625, Aug. 2023.
2. B. Panjwani et al., "Optimized machine learning for the early detection of polycystic ovary syndrome in women," *Sensors*, vol. 25, no. 4, p. 1166, Feb. 2025.
3. Z. Zad et al., "Predicting polycystic ovary syndrome with machine learning algorithms from electronic health records," *Front. Endocrinol.*, vol. 15, p. 1298628, Jan. 2024.
4. J. Kermanshahchi et al., "Development of a machine learning-based model for accurate detection and classification of polycystic ovary syndrome on pelvic ultrasound," *Cureus*, vol. 16, Aug. 2024.
5. M. M. Rahman et al., "Empowering early detection: A web-based machine learning approach for PCOS prediction," *Inform. Med. Unlocked*, vol. 47, 2024.
6. S. Arora, Vedpal, and N. Chauhan, "Polycystic ovary syndrome (PCOS) diagnostic methods in machine learning: a systematic literature review," *Multimed. Tools Appl.*, 2024. doi: 10.1007/s11042-024-19707-6
