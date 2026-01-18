import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("data.csv")

df.drop("id", axis=1, inplace=True)
df.drop("Unnamed: 32", axis=1, inplace=True)

df["area_perimeter_ratio"] = df["area_mean"] / df["perimeter_mean"]
df["area_concave_points_ratio"] = df["area_mean"] / df["concave points_mean"]
df["texture_smoothness_prod"] = df["texture_mean"] * df["smoothness_mean"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df["area_perimeter_ratio"].fillna(df["area_perimeter_ratio"].median(), inplace=True)
df["area_concave_points_ratio"].fillna(df["area_concave_points_ratio"].median(), inplace=True)

df["radius_bin"] = pd.qcut(df["radius_mean"], q=3, labels=[0, 1, 2])
df["texture_bin"] = pd.qcut(df["texture_mean"], q=3, labels=[0, 1, 2])

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

num_cols = X.columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
svc = SVC(kernel="rbf", probability=True, random_state=42)

estimators = [
    ("lr", lr),
    ("rf", rf),
    ("gb", gb),
    ("knn", knn),
    ("svc", svc)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", stacking_clf)
])

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

with open("stacking_pipeline.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)
