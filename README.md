# Power System Load Type Prediction (Assignment 2)

A machine learning project to classify industrial power system load into **Light Load**, **Medium Load**, and **Maximum Load** using historical consumption and operational features.

---

## 1. Project Objective

In electrical power systems and smart grid operations, accurate estimation of load demand is critical for generation scheduling, grid stability, energy dispatching, and dynamic pricing. The primary goal of this project is to build an explainable, beginner-friendly, and submission-ready machine learning model that predicts **`Load_Type`**:
- **`Light_Load`**: Off-peak periods with minimal power consumption (overnight hours, non-operating shifts, and weekends).
- **`Medium_Load`**: Normal operating periods characterized by standard industrial machinery operation.
- **`Maximum_Load`**: Peak demand periods when heavy machinery and production lines operate at maximum capacity.

---

## 2. Dataset Overview

The dataset is located in `data/load_data.csv` and contains 35,041 observations recorded at continuous 15-minute intervals across the full calendar year of 2018 (365 days).

### Column Description and Mapping

| Problem Statement Name | Actual Dataset Column Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Date` | `Date_Time` | String / Datetime | Timestamp in `DD-MM-YYYY HH:MM` format (15-min intervals) |
| `Usage_kWh` | `Usage_kWh` | Float | Industry energy consumption in kWh |
| `Lagging Current` | `Lagging_Current_Reactive.Power_kVarh` | Float | Reactive power for lagging current (kVarh) |
| `Leading Current` | `Leading_Current_Reactive_Power_kVarh` | Float | Reactive power for leading current (kVarh) |
| `CO2` | `CO2(tCO2)` | Float | Carbon dioxide emissions (tons of CO2) |
| *(Additional)* | `Lagging_Current_Power_Factor` | Float | Power factor for lagging current (%) |
| *(Additional)* | `Leading_Current_Power_Factor` | Float | Power factor for leading current (%) |
| `NSM` | `NSM` | Float | Number of Seconds from Midnight (0 to 86,400 s) |
| `Load Type` *(Target)* | `Load_Type` | String / Categorical | 3 classes: `Light_Load`, `Medium_Load`, `Maximum_Load` |

---

## 3. Machine Learning Approach

- **Model**: `DecisionTreeClassifier(random_state=42)` from `scikit-learn`.
- **Why Decision Trees?**:
  - Highly interpretable and explainable for industrial power engineers.
  - Generates clear decision thresholds on energy consumption and operating hours.
  - Invariant to monotonic feature scaling (no complex scaling pipelines needed).
  - Handles non-linear relationships and feature interactions natively.
- **Baseline Comparison**: `DummyClassifier(strategy='most_frequent')` is used as a naive baseline that always predicts the majority class (`Light_Load`).

---

## 4. Chronological Last-Month Evaluation Strategy

As explicitly specified in the problem statement, we evaluate performance using **the chronological last month as the held-out test set**:
- **Training Period**: January 1, 2018 to November 30, 2018 (11 months, 32,064 records, ~91.5% of data).
- **Testing Period**: December 1, 2018 to December 31, 2018 (1 month, 2,976 records, ~8.5% of data).
- **Zero Data Leakage**:
  - No random shuffling (`train_test_split(shuffle=True)` is strictly avoided).
  - Training dates strictly precede all testing dates (`train_max < test_min`).
  - Missing value imputation medians are calculated **strictly on the training set** and then applied to both train and test sets.

---

## 5. Evaluation Metrics & Experimental Results

Evaluated on the held-out December 2018 test set:

| Evaluation Metric | Dummy Baseline (`most_frequent`) | Decision Tree Classifier |
| :--- | :---: | :---: |
| **Accuracy** | 58.60% | **75.87%** |
| **Macro Precision** | 19.53% | **66.12%** |
| **Macro Recall** | 33.33% | **67.10%** |
| **Macro F1-Score** | 24.63% | **66.51%** |

### Detailed Decision Tree Classification Report (December Test Set)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`Light_Load`** | **95.72%** | **89.74%** | **92.63%** | 1,744 |
| **`Medium_Load`** | **53.54%** | **59.09%** | **56.18%** | 704 |
| **`Maximum_Load`** | **49.11%** | **52.46%** | **50.73%** | 528 |
| **Accuracy** | — | — | **75.87%** | 2,976 |
| **Macro Avg** | **66.12%** | **67.10%** | **66.51%** | 2,976 |
| **Weighted Avg** | **77.47%** | **75.87%** | **76.57%** | 2,976 |

### Confusion Matrix (December Test Set)

```text
                  Predicted Light   Predicted Medium   Predicted Maximum
Actual Light:          1,565              155                 24
Actual Medium:           25               416                263
Actual Maximum:          45               206                277
```

### Key Findings & Limitations
1. **Light Load Detection**: Nearly flawless identification (95.7% precision, 89.7% recall) due to distinct overnight low power consumption.
2. **Medium vs. Maximum Load**: The primary source of misclassification is boundary overlap between medium and maximum load during midday manufacturing hours.
3. **Feature Drivers**: Diurnal time features (`hour` at 47.2% and `NSM` at 14.7%) contributed over 60% of total decision importance, showing that load types are closely tied to scheduled factory shifts.
4. **Seasonality Limitation**: December represents winter operational conditions where heating and holiday slowdowns may differ from summer load patterns.

---

## 6. Project Structure

```text
ass-2ur/
├── Problem statement.docx        # Original assignment problem statement
├── data/
│   └── load_data.csv            # Original 15-minute industrial load dataset
├── Load_Type_Prediction.ipynb   # Complete, executed, submission-ready notebook
├── README.md                    # Project documentation & experimental report
└── requirements.txt             # Required Python package dependencies
```

---

## 7. How to Run the Notebook

### Step 1: Clone or Open the Project Directory
```bash
cd ass-2ur
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Open and Run in Jupyter
```bash
jupyter notebook
```
Open **`Load_Type_Prediction.ipynb`** and select **Kernel -> Restart & Run All**. All cells will execute in order and reproduce all figures, tables, and metrics.
