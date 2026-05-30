import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("train.csv")

print(df.head())
print(df.shape)

df.drop("Loan_ID", axis=1, inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
    
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print("Features:", X.columns)   # IMPORTANT for prediction check

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(importance)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.show()

# Create empty row with correct columns
new_applicant = pd.DataFrame(columns=X.columns)

# Fill values properly (MATCH ALL 11 FEATURES)
new_applicant.loc[0] = [
    1,    # Gender
    1,    # Married
    0,    # Dependents
    0,    # Education
    0,    # Self_Employed
    5000, # ApplicantIncome
    2000, # CoapplicantIncome
    150,  # LoanAmount
    360,  # Loan_Amount_Term
    1,    # Credit_History
    2     # Property_Area
]

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
    
    
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")