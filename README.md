# 🛡️ ChurnGuard AI: End-to-End Predictive Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Snowflake](https://img.shields.io/badge/Data_Warehouse-Snowflake-blue)
![Databricks](https://img.shields.io/badge/Compute-Databricks-orange)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project implements a robust, enterprise-grade pipeline to predict customer attrition. By leveraging a modern data stack (**Snowflake + Databricks**), the project transforms raw data into actionable insights via a machine learning model deployed through a user-friendly **Streamlit** interface.

---

## ⚙️ The Data Pipeline & Workflow

### 1. Data Ingestion & Warehouse (Snowflake)
* **Source:** Historical customer data (Telco Churn).
* **Storage:** Data was staged and loaded into **Snowflake** for centralized management.
* **Processing:** Performed SQL-based data cleaning, including schema standardization and handling of data types to ensure high integrity before the modeling phase.

### 2. Analytics & Modeling (Databricks)
* **Environment:** Leveraged **Databricks** clusters for high-performance computation.
* **Exploratory Data Analysis (EDA):** Conducted deep-dive analysis on churn drivers (Tenure, Contract type, Monthly charges).
* **Model Selection:** Evaluated multiple architectures, including:
    * Logistic Regression
    * XGBoost
    * **Random Forest (Champion Model)**
* **Optimization:** Used `GridSearchCV` with 5-fold cross-validation to maximize the **AUC-ROC** score.

### 3. Model Performance (Random Forest)
The model was tuned using `Entropy` criterion and a `max_depth` of 7 to prevent overfitting while maintaining high recall for the churn class.

| Metric | Score |
| :--- | :--- |
| **Best CV AUC** | **0.8509** |
| **Test Accuracy** | **80%** |
| **Test AUC Score** | **0.8281** |

**Confusion Matrix (Test Data):**
* **True Negatives:** 942 (Correctly predicted "Stay")
* **True Positives:** 188 (Correctly predicted "Churn")

---

## 🚀 Deployment (Streamlit)
While the model was managed via **Databricks Experiments (MLflow)**, the final application is deployed using **Streamlit** to allow business stakeholders to:
1. **Single Prediction:** Input individual customer details to see risk probability.
2. **Batch Prediction:** Upload a `.csv` file to analyze thousands of customers at once and download a report.

---

## 📁 Folder Structure
```text
Churn_Prediction
├── data/
│   └── sample.csv              # 5-record sample for testing
├── main/
│   └── churn_prediction.ipynb   # Databricks / Jupyter Notebook
├── app.py                      # Streamlit Application code
├── random_forest_model.pkl     # Optimized Pickle file
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
## 🛠️ Installation & Usage

Clone the repo:


git clone [https://github.com/PASUPULASUNIL/CHURN-GUARD--AI.git](https://github.com/PASUPULASUNIL/churn-guard--ai.git)

Install dependencies:

```pip install -r requirements.txt ```
Run the App:

``` streamlit run app.py ```

## 🔑 Key Takeaways

Full-Stack ML: Experience managing data from Cloud Warehouse (Snowflake) to Interactive UI (Streamlit).

Optimization: Successfully handled class imbalance to achieve a strong AUC of 0.82.

Scalability: The architecture is designed to handle batch processing for enterprise-level datasets.
