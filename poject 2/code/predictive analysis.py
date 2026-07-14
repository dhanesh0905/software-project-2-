import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score, roc_curve,
                              ConfusionMatrixDisplay, classification_report)
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

df = pd.read_csv("../data/titanic.csv")
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
df.head()


print("Missing values before cleaning:\n", df.isnull().sum())

df["age"] = df.groupby(["pclass", "sex"], observed=True)["age"].transform(
    lambda s: s.fillna(s.median())
)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print("\nMissing values after cleaning age/embarked:\n", df[["age", "embarked"]].isnull().sum())


dupes = df.duplicated().sum()
print(f"Duplicate rows found: {dupes} out of {len(df)} ({dupes/len(df):.1%})")
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after de-duplication:", df.shape)


drop_cols = ["alive", "class", "who", "adult_male", "embark_town", "deck", "alone"]
df = df.drop(columns=drop_cols)
print("Dropped columns:", drop_cols)
print("Remaining columns:", list(df.columns))

df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["embarked"], prefix="embarked", drop_first=True)
df.head()

print("Overall survival rate: {:.1%}".format(df["survived"].mean()))
print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean().rename({0: "male", 1: "female"}))
print("\nSurvival rate by passenger class:\n", df.groupby("pclass")["survived"].mean())
print("\nSurvival rate by sibsp (siblings/spouses aboard):\n", df.groupby("sibsp")["survived"].mean())


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=axes[0, 0])
axes[0, 0].set_title("Survival Rate by Class and Sex")
axes[0, 0].legend(title="sex", labels=["male", "female"])

sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True, ax=axes[0, 1],
             palette=["#C44E52", "#55A868"])
axes[0, 1].set_title("Age Distribution by Survival")

sns.boxplot(data=df, x="survived", y="fare", ax=axes[1, 0])
axes[1, 0].set_title("Fare by Survival Outcome")
axes[1, 0].set_ylim(0, 300)

corr = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], cbar=False)
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("../outputs/eda_overview.png", bbox_inches="tight")
plt.show()
print("Saved eda_overview.png")

feature_cols = [c for c in df.columns if c not in ("survived",)]
X = df[feature_cols]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, max_depth=6),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

results_df = pd.DataFrame({k: {m: round(v, 3) for m, v in r.items() if m not in ("y_pred", "y_proba")}
                            for k, r in results.items()}).T
print(results_df)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Did not survive", "Survived"]).plot(
        ax=axes[i], colorbar=False, cmap="Blues"
    )
    axes[i].set_title(f"{name}\nConfusion Matrix")

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.2f})")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[2].set_title("ROC Curves")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend()

plt.tight_layout()
plt.savefig("../outputs/model_evaluation.png", bbox_inches="tight")
plt.show()
print("Saved model_evaluation.png")

best_model_name = results_df["roc_auc"].idxmax()
print(f"Best model by ROC-AUC: {best_model_name}")
print("\nClassification report (Random Forest):\n",
      classification_report(y_test, results["Random Forest"]["y_pred"],
                             target_names=["Did not survive", "Survived"]))

rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Feature importances (Random Forest):\n", importances)

plt.figure(figsize=(7, 4.5))
importances.plot(kind="barh", color="#4C72B0")
plt.gca().invert_yaxis()
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../outputs/feature_importance.png", bbox_inches="tight")
plt.show()
print("Saved feature_importance.png")

test_results = X_test.copy()
test_results["actual_survived"] = y_test.values
test_results["predicted_survived"] = results["Random Forest"]["y_pred"]
test_results["predicted_probability"] = results["Random Forest"]["y_proba"].round(3)
test_results.to_csv("../outputs/test_set_predictions.csv", index=False)
print("Saved test set predictions to outputs/test_set_predictions.csv")
print("\nProject 2 complete.")
