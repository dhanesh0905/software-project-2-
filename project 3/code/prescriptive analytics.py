import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import linprog
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

df = pd.read_csv("../data/car_crashes.csv")
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
df.head()

print("Missing values per column:\n", df.isnull().sum())
print("\nData types look correct:", df.dtypes.to_dict())
print("\nSanity check - any negative percentages/rates?",
      (df.select_dtypes("number") < 0).any().any())

dupes = df.duplicated().sum()
print(f"Duplicate rows found: {dupes}")
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after de-duplication:", df.shape)

identifier_col = "abbrev"
model_feature_cols = ["speeding", "alcohol", "not_distracted", "no_previous"]
print("Identifier (excluded from modeling):", identifier_col)
print("Modeling features:", model_feature_cols)

region_map = {
    **{s: "Northeast" for s in ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"]},
    **{s: "Midwest" for s in ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"]},
    **{s: "South" for s in ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY",
                             "MS", "TN", "AR", "LA", "OK", "TX"]},
    **{s: "West" for s in ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"]},
}
df["region"] = df["abbrev"].map(region_map)
print("Any states unmapped to a region?", df["region"].isnull().sum())

region_dummies = pd.get_dummies(df["region"], prefix="region")
df = pd.concat([df, region_dummies], axis=1)
df[["abbrev", "region"] + list(region_dummies.columns)].head()

print(df[model_feature_cols + ["total", "ins_premium", "ins_losses"]].describe())

print("\nCorrelation with total crash rate:")
print(df[model_feature_cols + ["total"]].corr()["total"].sort_values(ascending=False))

print("\nAverage crash rate by region:")
print(df.groupby("region")["total"].mean().sort_values(ascending=False))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.scatterplot(data=df, x="alcohol", y="total", ax=axes[0, 0], s=50)
for _, row in df.iterrows():
    axes[0, 0].annotate(row["abbrev"], (row["alcohol"], row["total"]), fontsize=6, alpha=0.6)
axes[0, 0].set_title("Alcohol Involvement vs. Total Crash Rate")

sns.scatterplot(data=df, x="speeding", y="total", ax=axes[0, 1], s=50, color="#DD8452")
axes[0, 1].set_title("Speeding Involvement vs. Total Crash Rate")

top10 = df.nlargest(10, "total").sort_values("total")
axes[1, 0].barh(top10["abbrev"], top10["total"], color="#C44E52")
axes[1, 0].set_title("Top 10 States by Crash Rate")
axes[1, 0].set_xlabel("Crashes per billion miles")

sns.boxplot(data=df, x="region", y="total", ax=axes[1, 1])
axes[1, 1].set_title("Crash Rate by Region")

plt.tight_layout()
plt.savefig("../outputs/eda_overview.png", bbox_inches="tight")
plt.show()
print("Saved eda_overview.png")

X = df[model_feature_cols]
y = df["total"]

lr = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
r2_scores = cross_val_score(lr, X, y, cv=cv, scoring="r2")
rmse_scores = -cross_val_score(lr, X, y, cv=cv, scoring="neg_root_mean_squared_error")

print(f"Cross-validated R^2:   mean={r2_scores.mean():.3f}, scores={np.round(r2_scores, 3)}")
print(f"Cross-validated RMSE:  mean={rmse_scores.mean():.3f}, scores={np.round(rmse_scores, 3)}")

lr.fit(X, y)
coefs = pd.Series(lr.coef_, index=model_feature_cols)
print("\nFitted coefficients (full-data fit, for interpretation):\n", coefs)
print(f"Intercept: {lr.intercept_:.3f}")

actionable_model = LinearRegression().fit(df[["speeding", "alcohol"]], df["total"])
actionable_coefs = pd.Series(actionable_model.coef_, index=["speeding", "alcohol"])
print(f"Dedicated 2-variable model R^2: {actionable_model.score(df[['speeding','alcohol']], df['total']):.3f}")
print("Coefficients (both positive, as expected):\n", actionable_coefs)

df["actionable_risk_score"] = (actionable_coefs["speeding"] * df["speeding"]
                                + actionable_coefs["alcohol"] * df["alcohol"])
df["campaign_cost"] = df["ins_premium"]  # simplifying cost-proxy assumption, stated above

total_budget = 0.30 * df["campaign_cost"].sum()
print(f"Total cost to fund every state fully: {df['campaign_cost'].sum():,.0f}")
print(f"Available budget (30% of that):       {total_budget:,.0f}")

n = len(df)
c = -df["actionable_risk_score"].values
A_ub = [df["campaign_cost"].values]
b_ub = [total_budget]
bounds = [(0, 1)] * n

lp_result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
df["funding_fraction_LP"] = lp_result.x
print(f"\nLP solver status: {lp_result.message}")
print(f"Optimal total expected benefit: {-lp_result.fun:.2f}")

df["benefit_per_dollar"] = df["actionable_risk_score"] / df["campaign_cost"]
greedy_order = df.sort_values("benefit_per_dollar", ascending=False).copy()
remaining_budget = total_budget
greedy_fraction = pd.Series(0.0, index=greedy_order.index)
for idx, row in greedy_order.iterrows():
    if remaining_budget <= 0:
        break
    take_cost = min(row["campaign_cost"], remaining_budget)
    greedy_fraction[idx] = take_cost / row["campaign_cost"]
    remaining_budget -= take_cost
df["funding_fraction_greedy"] = greedy_fraction
agreement = np.allclose(df["funding_fraction_LP"], df["funding_fraction_greedy"], atol=1e-4)
print(f"Greedy heuristic matches LP solution: {agreement}")

recommendation = df[["abbrev", "region", "total", "actionable_risk_score", "campaign_cost",
                     "benefit_per_dollar", "funding_fraction_LP"]].sort_values(
    "funding_fraction_LP", ascending=False
)
fully_funded = recommendation[recommendation["funding_fraction_LP"] > 0.999]
partially_funded = recommendation[(recommendation["funding_fraction_LP"] > 0.001) &
                                   (recommendation["funding_fraction_LP"] <= 0.999)]
not_funded = recommendation[recommendation["funding_fraction_LP"] <= 0.001]

print(f"Fully funded states ({len(fully_funded)}): {list(fully_funded['abbrev'])}")
print(f"Partially funded state ({len(partially_funded)}): {list(partially_funded['abbrev'])}")
print(f"Not funded this round ({len(not_funded)} states)")
recommendation.head(15)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_df = recommendation.head(15).iloc[::-1]
colors = ["#55A868" if f > 0.999 else ("#DD8452" if f > 0.001 else "#C44E52")
          for f in plot_df["funding_fraction_LP"]]
axes[0].barh(plot_df["abbrev"], plot_df["funding_fraction_LP"], color=colors)
axes[0].set_title("Prescribed Funding Fraction — Top 15 States")
axes[0].set_xlabel("Fraction of full funding")
axes[0].set_xlim(0, 1.05)

axes[1].scatter(df["campaign_cost"], df["actionable_risk_score"],
                 c=df["funding_fraction_LP"], cmap="RdYlGn", s=60, edgecolor="k", linewidth=0.3)
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
plt.colorbar(sm, ax=axes[1], label="Funding fraction")
axes[1].set_xlabel("Campaign cost (insurance-premium proxy)")
axes[1].set_ylabel("Actionable risk score")
axes[1].set_title("Cost vs. Benefit, colored by Funding Decision")

plt.tight_layout()
plt.savefig("../outputs/prescriptive_allocation.png", bbox_inches="tight")
plt.show()
print("Saved prescriptive_allocation.png")

recommendation.to_csv("../outputs/budget_allocation_recommendation.csv", index=False)
print("Saved outputs/budget_allocation_recommendation.csv")
print("\nProject 3 complete.")
