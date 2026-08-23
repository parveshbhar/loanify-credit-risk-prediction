# Loanify — Credit Risk & Loan Approval Prediction System

Loanify is a **supervised machine learning classification project** that predicts whether a loan application is likely to be approved based on applicant financial and demographic information.

The project uses **Logistic Regression** with data preprocessing, categorical encoding, feature engineering, feature scaling, and classification metrics.

---

## 📌 Project Overview

Loan approval decisions depend on multiple factors such as income, credit score, debt-to-income ratio, employment information, and other applicant attributes.

Loanify processes these features and uses a Logistic Regression model to predict the loan approval outcome.

### Key Objectives

* Clean and preprocess raw loan application data
* Handle missing values
* Convert categorical data into numerical form
* Engineer additional financial features
* Scale numerical features
* Train a Logistic Regression classification model
* Evaluate model performance using multiple classification metrics

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine learning and preprocessing
* **Seaborn** — Data visualization

### Machine Learning Techniques

* Supervised Learning
* Binary Classification
* Logistic Regression
* Label Encoding
* One-Hot Encoding
* Feature Engineering
* Standard Scaling
* Train-Test Split

---

## 🔄 Machine Learning Workflow

```text
Raw Loan Dataset
       ↓
Data Loading
       ↓
Missing Value Handling
       ↓
Remove Applicant ID
       ↓
Label Encoding
       ↓
Feature Engineering
       ↓
One-Hot Encoding
       ↓
Train-Test Split
       ↓
Standard Scaling
       ↓
Logistic Regression
       ↓
Prediction
       ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

### 1. Missing Value Handling

Numerical columns with missing values are filled using the **mean** of the respective column.

Categorical columns with missing values are filled using the **mode**.

```python
df[numerical_columns] = df[numerical_columns].fillna(
    df[numerical_columns].mean()
)

df[categorical_columns] = df[categorical_columns].fillna(
    df[categorical_columns].mode().iloc[0]
)
```

### 2. Removing Applicant ID

`Applicant_ID` is removed because it is an identifier and does not provide meaningful predictive information.

```python
df = df.drop("Applicant_ID", axis=1)
```

### 3. Label Encoding

Selected categorical variables such as `Education_Level` and the target variable `Loan_Approved` are converted into numerical representations using `LabelEncoder`.

### 4. One-Hot Encoding

Remaining categorical input features are converted into binary numerical features using One-Hot Encoding.

```python
pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)
```

---

## ⚙️ Feature Engineering

Additional features were created to provide the model with transformed representations of important financial variables.

### DTI Ratio Squared

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
```

### Credit Score Squared

```python
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

These transformations allow the model to use additional information about the relationship between financial variables and loan approval.

---

## 🤖 Machine Learning Model

### Logistic Regression

Loanify uses **Logistic Regression** as its classification algorithm.

```python
model = LogisticRegression(
    solver="liblinear"
)

model.fit(
    X_train_scaled,
    y_train
)
```

The model predicts whether a loan application belongs to the approved or rejected class.

---

## 📊 Model Evaluation

The model was evaluated on a test set using an **80/20 train-test split**.

### Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **87.50%** |
| Precision | **79.03%** |
| Recall    | **80.33%** |
| F1-Score  | **79.67%** |

### Metrics Used

**Accuracy**
Measures the overall percentage of correct predictions.

**Precision**
Measures how many of the applications predicted as approved were actually approved.

**Recall**
Measures how many of the actual approved applications were correctly identified.

**F1-Score**
Provides a balance between Precision and Recall.

**Confusion Matrix**
Used to analyze correct and incorrect predictions across the two classes.

---

## 📁 Project Structure

```text
loanify-credit-risk-prediction/
│
├── loanify.py
├── loan_approval_data.csv
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/loanify-credit-risk-prediction.git
```

### 2. Navigate to the Project

```bash
cd loanify-credit-risk-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Model

```bash
python loanify.py
```

The program will train the Logistic Regression model and display the evaluation metrics.

---

## 📦 Requirements

The project requires:

```text
pandas
numpy
scikit-learn
seaborn
```

These dependencies are also provided in `requirements.txt`.

---

## 🔮 Future Improvements

* Compare Logistic Regression with **Naive Bayes, Random Forest, and other classification algorithms**
* Perform **hyperparameter tuning**
* Implement **cross-validation**
* Add **ROC-AUC evaluation**
* Improve feature selection
* Create a user-friendly prediction interface using **Streamlit**
* Deploy the trained model as a web application or API

---

## 🎯 Key Learning Outcomes

Through this project, I practiced:

* Data cleaning and preprocessing
* Missing-value handling
* Categorical feature encoding
* Feature engineering
* Feature scaling
* Supervised machine learning
* Binary classification
* Logistic Regression
* Model evaluation
* Git and GitHub project management

---

## 👨‍💻 Author

**Parvesh Bhar**

GitHub: `https://github.com/parveshbhar`
