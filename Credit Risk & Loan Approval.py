import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score ,precision_score,recall_score,f1_score,confusion_matrix)

df = pd.read_csv("loan_approval_data.csv")

# Missing Value Handling

categorical_columns = df.select_dtypes(include=["object", "string"]).columns

numerical_columns = df.select_dtypes( include=["number"]).columns

df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

df[categorical_columns] = df[categorical_columns].fillna( df[categorical_columns].mode().iloc[0])

df = df.drop("Applicant_ID", axis=1)


le = LabelEncoder()

df["Education_Level"] = le.fit_transform(df["Education_Level"])

df["Loan_Approved"] = le.fit_transform( df["Loan_Approved"])

# Feature Engineering

df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2

df["Credit_Score_sq"] = df["Credit_Score"] ** 2

X = df.drop(columns=["Loan_Approved","Credit_Score","DTI_Ratio"])

y = df["Loan_Approved"]

# One-Hot Encoding

categorical_columns = X.select_dtypes(include=["object", "string"]).columns

X = pd.get_dummies(X,columns=categorical_columns,drop_first=True,dtype=int)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(solver="liblinear")
model.fit(X_train_scaled,y_train)
y_pred = model.predict( X_test_scaled)

print( "Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
