# AI Text Detection Project - Development Log

**Student:** Hannah Soliao
**Project:** AI Text Detection for Academic Integrity
**Timeline:** December 18, 2025 - January 10, 2026

## Week 1: Foundation & Data Exploration

###Day 1 - December 18, 2025
**Goal:** Project setup and environment configuration

**Completed:**

- Created project directory structure
- Set up VS Code with Jupyter extension
- Created Python virtual environment
- Installed initial packages (pandas,numpy,matplotlib,seaborn,scikit-learn)
- Initialized Git repository
- Created README.md with project information
- Created all project folders with .gitkeep files

**Files Created:**

- Project structure (all folders)
- README.md
- .gitignore
- requirements.txt (initial)

**Time Spent:** 3 hours

**Notes:** Development environment ready. All foundational structure in place.

---

### Day 2 - December 19, 2025

**Goal:** Initial data exploration and understanding

**Completed:**

- Download AI Text Detection Pile dataset from Hugging Face (~1.3M samples)
- Created 01_data_exploration.ipynb notebook
- Converted dataset to pandas DataFrame
- Created stratified 10% sample (df_sample) for exploration
- Examine basic structure: 2 columns (text, source), no missing values
- Analyzed class distribution: ~73.82% human and ~26.18% AI **(2.82:1 imbalance ratio)**
- Generated visualizatins:
  - Class distributions bar charts (counts and percetages)
  - Text length distributions (histograms, box plots, density plots)
  - Word clouds for both human and AI texts
- Documented initial observations and patterns

**Key Findings:**

- Significant class imbalance (2.82:1) will require handling during modeling
- Text length shows human texts are longer than AI. However, AI text has more variations in length compared to human text.
- No data qulity issues detected
- Sample successully maintains original proportions

**Files Created:**

- notebooks/01_data_exploration.ipynb
- data/raw/ai_text_detection_full.parquet (saved for reuse)
- visualizations/plots/class_distribution.png
- visualizations/plots/text_length_distributions.png
- visualizations/plots/wordcloud_human.png
- visualizations/plots/wordcloud_ai.png

**Time Spent:** 5 hours

**Next Steps:** Deep dataset analysis and quality assessment

---

### Day 3 - December 20, 2025

**Goal:** Deep dataset analysis and quality assessment

**Planned:**

- Analyze text characteristics (sentenc length, word length, vocabulary diversity)
- Check for duplicates and outliers
- Create comparitive visualizations
- Verify label quality (spot-check samples)

**Status:** Not Started

### Day 4-5 - December 21-22, 2025 (Weeken)

**Goal:** Background research and planning refinement

**Planned:**

- Research existing AI detection methods (Turnitin, GPTZEro)
- Find academic papers on AI text detection
- Refine model comparison strategy
- Plan evaluation metrics approach

**Status:** Not Started

---

### Day 6 - December 23, 2025

**Goal:** Begin data preprocessing

**Planned:**

- Create 02_data_preprocessing.ipynb
- Handle class imbalance
- Clean text data
- Create train/validation/test splits (70/15/15)

**Status:** Not Started

---

### Day 7 - December 24, 2025

**Goal:** Create reusable preprocessing functions

**Planned:**

- Create src/data_preprocessing.py module
- Write reusable functions with docstrings
- Refactor notebook to use module

**Status:** Not Started

---

## Week 2: Feature Engineering & Baseline Models

Not Started

---

## Key Decisions Made

1. **Sampling Strategy:** Using 10% stratified sample for exploration/development, full dataset for final training
2. **Data Format:** Saved as parquet for efficiency
3. **Visualization Style:** Professional formatting with clear labels and titles
4. **Documentation:** Extensive markdown cells in notebooks explaining reasoning

---

## Technical Notes

- Dataset: artem9k/ai-text-detection-pile from Hugging Face
- Sample: 10% stratified (maintain 2.82:1 class distribution)
- Visualization library: matplotlib + seaborn
- Development approach: Sample for exploration, full dataset for training

---

## Assessments Requirements Tracking

### Model Development (50%) - In Progress

- [ ] Multiple algorithms implemented
- [x] Dataset loaded and understood
- [ ] Feature engineering completed
- [ ] Comprehensive evaluation
- [ ] Results well-presented

### Technique (30%) - In Progress

- [x] Data exploration completed
- [ ] Data preprocessing implemented
- [ ] Multiple models compared
- [ ] Hyperparamter tuning
- [ ] Proper evaluation methodology

### Documentation (20%) - Not Started

- [x] Github repository structure created
- [x] README.md with identification
- [ ] 800-word technical document
- [ ] Critical reflection
- [ ] Proper terminology usage

---

**Last Updated:** December 19, 2025, 3:30am
