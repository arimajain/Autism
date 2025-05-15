# 🧠 Autism Prediction Using Machine Learning

Autism is a complex neurological disorder that affects a person's ability to interact, communicate, and behave. While traditional diagnostic methods for Autism Spectrum Disorder (ASD) are subjective and often inaccessible, this project demonstrates how machine learning can be leveraged to predict whether an individual is likely to have autism based on various behavioral and demographic features.

## 📌 Project Overview

This project applies data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning classification models to predict autism. It showcases the end-to-end pipeline from raw data ingestion to building and evaluating predictive models.

## 📂 Dataset

- Source: `train.csv`  
- Rows: 800  
- Columns: 22  
- Features include behavioral scores (A1–A10), demographics (age, gender, ethnicity), and medical history (jaundice, family history).

## 🛠️ Tech Stack

- **Languages**: Python  
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `scikit-learn`, `xgboost`
  - Imbalanced Learning: `imblearn`

## 🧹 Step 1: Data Cleaning

- Handled inconsistent values like `'others'`, `'Others'`, and `'?'`.
- Converted categorical labels (`'yes'`, `'no'`) into binary format (`1`, `0`).
- Removed outliers from continuous features (`result` column).

## 📊 Step 2: Exploratory Data Analysis

- **Class Imbalance**: Detected severe imbalance between ASD and non-ASD classes.
- **Visualizations**:
  - Count plots for categorical and integer features.
  - Distribution plots for numerical data.
  - Heatmap to identify multicollinearity.
- Key Insights:
  - Gender and country have varied distributions.
  - Higher `sum_score` (A1 to A10) is strongly correlated with autism.

## 🧪 Step 3: Feature Engineering

- Created new features:
  - `ageGroup`: Binned age into groups like Toddler, Kid, Teenager, etc.
  - `sum_score`: Total score across A1 to A10 behavioral indicators.
  - `ind`: Combined medical history indicators (`austim`, `used_app_before`, `jaundice`).
- Applied log transformation to normalize skewed age distribution.
- Encoded categorical features using `LabelEncoder`.

## 🤖 Step 4: Model Training

- **Preprocessing**:
  - Dropped irrelevant or highly correlated features (`ID`, `age_desc`, etc.).
  - Applied oversampling using `RandomOverSampler` to handle class imbalance.
  - Scaled features using `StandardScaler`.

- **Models Trained**:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - XGBoost Classifier

- **Evaluation Metric**: ROC-AUC Score

| Model              | Training AUC | Validation AUC |
|-------------------|--------------|----------------|
| LogisticRegression| High         | Good           |
| XGBoostClassifier | Very High    | Overfit        |
| SVC               | High         | Good           |

✅ **Best Performing Models**: Logistic Regression & SVC (least overfitting)

## 🎯 Conclusion

- Machine learning models can aid in predicting autism, especially when traditional diagnostic tools are unavailable or inaccessible.
- Feature engineering (like `sum_score`) plays a key role in boosting performance.
- Class imbalance significantly affects performance and needs to be addressed for robust predictions.
