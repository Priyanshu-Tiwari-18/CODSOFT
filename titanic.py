import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)

df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df["AgeGroup"] = pd.cut(df["Age"], [0,12,18,35,60,100],
                        labels=["Child","Teen","Adult","Middle","Senior"])
df["FareGroup"] = pd.cut(df["Fare"], [0,7.9,14.5,31,100,1000],
                         labels=["Low","Medium","High","VeryHigh","Luxury"])

df["Sex"] = df["Sex"].map({"male":1, "female":0})
df = pd.get_dummies(df, columns=["Embarked","AgeGroup","FareGroup"], drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv = cross_val_score(model, X_train, y_train, cv=5)
print("CV Mean:", round(cv.mean(), 4))

feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Feature Importance")
plt.show()

joblib.dump(model, "titanic_model.pkl")
print("Model saved as titanic_model.pkl")
