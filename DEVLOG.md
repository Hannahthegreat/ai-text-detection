# AI Text Detection Project - Development Log

**Student:** Hannah Soliao
**Project:** AI Text Detection for Academic Integrity
**Timeline:** December 18, 2025 - January 10, 2026

---

## Phase 1: Foundation & Data Exploration

### Week 1 Highlights

- **Environment Setup** - VS Code, Python virtual environment, inital dependencies
- **Dataset Acquisition** - Downloaded AI Text Detection Pile (1.39M samples)
- **Initial Analysis** - Identified 2.82:1 class imblance, analyzed text characteristics
- **Key Decision:** Stratified 30% sampling strategy for computational constraints

**Deliverables:** Project structure, data exploration notebook, initial visualizations

---

## Phase 2: Data Processing & Feature Engineering (Dec 25-31)

### Week 2 Highlights

- **Data Preprocessing** - Text cleaning, URL removal, normalization
- **Train/Val/Test Split** - 70-15-15 stratified split (292K/63K/63K samples)
- **Feature Engineering** - TF-IDF vectorization (5000 features, unigrams + bigrams)
- **Key Learning:** Class imblanace requires cost-sensitive approaches

**Deliverables:** Cleaned datset, feature matrices, preprocessing modules

---

## Phase 3: Model Development (Jan 1-7)

### Jan 1-2: Baseline Models

- Implemented Logisttic Regression (84.2% pecision, 80.0% recall)
- Implemented Naive Bayes (88.8% precision, 39.4% recall)
- **Insight:** Tree-based models show competitive precison with varying recall

### Jan 5-6: Ensemble Experimentation

- Four-model ensemble (72.98% precision) - Naive Bayes contamination
- Two-model ensemble LR + XGBoost (85.16% precision, 81.24% recall)
- **Critical Learning:**j Strategic model selection > quantity

### Jan 7: Evaluatin & Analysis

- Comprehensive metrics: confusion matrices, ROC curves, PR curves
- Feature importance analysis
- Ethical trade-off documentation

**Deliverables:** Four trained models, evaluation framework, ensemble model

---

## Phase 4: Interface & Documentation (Jan 8-10)

### Jan 8: Web Interface

- Built Gradio application with custom CSS
- Implemented real-time text analysis
- Added example texts and statistis display

### Jan 9: Documentation & Polish

- Technical documentation (800 words)
- Repositary organizatin and README
- Code cleanup and commenting

### Jan 10: Final Submission

- Final testing and verification
- Documentation review
- Project submission

**Deliverables:** Web interface, complete documentation, polished repository

---

## Key Milestones

| Date   | Milestone                    | Status |
| ------ | ---------------------------- | ------ |
| Dec 18 | Project initialized          | ✅     |
| Dec 24 | Data exploration complete    | ✅     |
| Dec 31 | Feature engineering complete | ✅     |
| Jan 4  | All models trained           | ✅     |
| Jan 6  | Ensemble model finalized     | ✅     |
| Jan 8  | Web interface deployed       | ✅     |
| Jan 10 | Final submission             | ✅     |

---

## Technical Decisions Log

### Data Processing

- **Decision:** Use 30% stratified sample
- **Raitonale:** Balance between computational constraints and model performance
- **Outcome:** Maintained class distribution, enabled efficient iteration

### Feature Engineering

- **Decision:** TF-IDF with 5000 features, unigrams + bigrams
- **Raitonale:** Capture bothy individual words and phrases
- **Outcome:** Effective feature representation without excessive dimensionality

### Class Imbalance

- **Decision:** Cost-sensitive learning with classs weights
- **Rationale:** Preserve all data rather than undersampling
- **Outcome:** Better performance than balanced sampling approaches

### Model Selection

- **Decision:** LR + XGBoost ensemble only
- **Rationale:** Strategic selection based on complementary strengths
- **Outcome:** 85.16% precision vs 72.98% with all four models

### Precision Priority

- **Decision:** Optimize for precision over recall
- **Rationale:** False accusations more harmful in academic context
- **Outcome:** Ethical considerations integrated into model design

---

## Challenges & Solutions

### Challenge 1: Class Imbalance

- **Problem:** 2.82:1 ratio affecting model performance
- **Solution:** Cost-sensitive learning with calculated class weights
- **Learning:** Better than undersampling which loses informatino

### Challenge 2: Ensemble Performance

- **Problem:** Four-model ensemble worse than individual models
- **Solution:** Strategic two-model selection (LR + XGBoost)
- **Learning:** Model compatibilty matters more than quantity

### Challenge 3: Computational Constraints

- **Problem:** Mac unable to process 1.39M samples
- **Solution:** Stratified 30% sampling
- **Learning:** Smart sampling maintains statistical properties

### Challenge 4: Dataset Age

- **Problem:** Training data from 2021-2022, older AI patterns
- **Solution:** Documented as limitation, discussed modern AI evasion
- **Learning:** Model generalization limitations acknowledged

---

## Resulst Summary

### Final Model Performance

- **Precision:** 85.16% (minimize false accusaitons)
- **Recall:** 81.24% (reasonable detection rate)
- **F1-Score:** 83.15%
- **Accuracy:** 91.39%

### Comparison to Targets

- Precision > 80% achieved
- Professional-grade system built
- Ethical considerations integrated
- Web interface deployed
- Complete documentation

---

## Lessons Learend

1. **Precision Matters in Context** - Academic integrity requires prioritizing false positive avoidance
2. **Ensemble Intelligence** - Strategic seleciton beets "throw everything in" approach
3. **Dataset Age Matters** - AI text patterns evolve, model training data becomes outdated
4. **Cost-Sensitive Learning** - Effective for imbalanced classes without data loss
5. **Iterative Experimentation** - Testing multiple approahes reveals optimal solutions

---

## Files Delivered

- `01_data_exploration.ipynb` - Initial data analysis
- `02_data_preprocessing.ipynb` - Data Cleaning and splitting
- `03_model_training.ipynb` - Model Development and evaluation
- `04_ensemble_experiments.ipynb` - Weighting and Soft Voting ensemble experiments
- `05_final_model.ipynb` - Final LR + XGBoost model
- `app.py` - Gradio web interface
- `style.css` - Custom styling
- `README.md` - Project documentation
- `requirements.txt` - Dependencies
- `models/saved_models/` - Trained models (LR, XGBoost, ensemble)

---

## Time Investment

- **Setup & Exploration:** 8 hours
- **Data Processing:** 12 hours
- **Model Development:** 25 hours
- **Interface & Documentation:** 8 hours
- **Testing & Polish:** 8 hours

**Total:** ~61 hours over 24 days

---

**Project Status:** Complete and submitted January 10, 2026
