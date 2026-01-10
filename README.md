# AI Text Detection for Academic Integrity

- **Student:** Hannah Soliao
- **Registration:** 639188
- **Module:** MACHINE LEARNING (CPU6100-20)
- **Program:** Bsc (Hons) Creative Computing @ Academic Centre RAK, UAE

## Project Overview

This project develops a machine learning system for detecting AI-generated text for academic integrity purposes. Built to address accuracy concenrs in Learning Management Systems (LMS) like Turnitin and Canvas, this project prioritizes precision to minimize false accusations against students.

**Key Achivement:** Ensemble model (LR + XGBoost) with 85.16% precision, 81.24% recall, and 91.39% accuracy.

## Problem Context

Educational institutions increasingly face challenges with AI-generated academic submissions. Existing detection tools can proudce false positives that damage student reputations and academic standing. This project explores machine learning approaches that balance detection acccuray with ethical considerations.

## Dataset

[Hugging Face AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) - 1.39M samples with 2.82:1 class imbalance (human:AI ratio)

**Preprocessing**

- Stratified 30% sample (417,757 instances) for computational efficiency
- 70-15-15 split (train/validation/test)
- Text cleaning and normalization

## Models Implemented

1. **Logistic Regression** - 84.2% precision, 80.0% recall
2. **XGBoost** - 85.8% precision, 74.8% recall
3. **Random Forest** - 84.7% precision, 59.6% recall
4. **Naive Bayes** - 88.8% precision, 39.4% recall

**Final Ensemble (LR + XGBoost):** 85.1% precision, 81.24% recall, 91.39% accuracy

## Features

- TF-IDF vectorization (5000 features, unigrams + bigrams)
- Cost-sensitive learning with class weights
- Ensemble learning with strategic model selection
- Web interface with Gradio

## Installation

```bash
# Clone repository
git clone []


# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
# Run notebooks in order:
# 01_data_exploration.ipynb
# 02_data_preprocessing.ipynb
# 03_model_training.ipynb
# 03_model_training.ipynb
# 05_final_model.ipynb
```

### Web Interface

```bash
python app.py
```

Access at `http://127.0.0.1:7860`

## Project Structure

```
ai-text-detection/
├── notebooks/           # Jupyter notebooks for analysis
├── models/             # Saved model files
├── app.py              # Gradio web interface
├── static/             # CSS styling
├── requirements.txt    # Dependencies
└── README.md
```

## Results

| Model               | Precision | Recall | F1-Score | Accuracy |
| ------------------- | --------- | ------ | -------- | -------- |
| LR + XGBoost        | 85.16%    | 81.24% | 83.15%   | 91.39%   |
| Logistic Regression | 84.2%     | 80.0%  | 82.0%    | 90.9%    |
| XGBoost             | 85.8%     | 74.8%  | 79.9%    | 90.5%    |

**Key Insight:**. Precision prioritized over recall to minimize false accusations. Two-model ensemble outperformed four-model approach due to strategic selection.

## Ethical Considerations

In academic contexts, false positives (wrongly accused students) cause more harm than false negatives (missing AI text). The model design explicityly prioritizes precision to minimize reputational damage while maintaining reasonable detection capability.

**Limitations:**

- Dataset contains older AI text patterns (2021-2022)
- Modern AI models may evade detection
- Effectiveness varies with text length and domain
- Should be used as supportive tool, not definitive judgement

## Future Improvements

1. Fine-tuning on recent AI model outputs
2. Domain-specific adaptation for different disciplines
3. Confidence calibration for uncertainty quantification
4. Expanded feature engineering (semantic, syntatic)

---

This project was completed independently as part of Assessment 2 for CPU-6100-20 Machine Learning at Academic Center RAK, UAE.
