import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

df = pd.read_csv("../data/diamonds.csv")
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
df.head()

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Missing values per column:\n", df.isnull().sum())
print("\nRows with impossible zero dimensions (x, y, or z == 0):",
      ((df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0)).sum())


before = len(df)
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
print(f"Removed {before - len(df)} rows with impossible dimensions.")

dupes = df.duplicated().sum()
print(f"Duplicate rows found: {dupes}")
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after de-duplication:", df.shape)

df["volume"] = df["x"] * df["y"] * df["z"]
print("Added derived 'volume' feature (x * y * z) to summarize the 3 dimension columns.")
df[["x", "y", "z", "volume", "carat"]].describe()

cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]              # J worst -> D best
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # worst -> best

df["cut_encoded"] = df["cut"].apply(lambda v: cut_order.index(v))
df["color_encoded"] = df["color"].apply(lambda v: color_order.index(v))
df["clarity_encoded"] = df["clarity"].apply(lambda v: clarity_order.index(v))

df[["cut", "cut_encoded", "color", "color_encoded", "clarity", "clarity_encoded"]].sample(5, random_state=RANDOM_STATE)

print(df[["carat", "depth", "table", "price"]].describe())

print("\nAverage price by cut:")
print(df.groupby("cut", observed=True)["price"].mean().sort_values())

print("\nAverage price by clarity:")
print(df.groupby("clarity", observed=True)["price"].mean().sort_values())

print("\nCorrelation of price with numeric features:")
print(df[["carat", "depth", "table", "volume", "cut_encoded", "color_encoded",
          "clarity_encoded", "price"]].corr()["price"].sort_values(ascending=False))


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.histplot(df["price"], bins=50, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Distribution of Price")
axes[0, 0].set_xlabel("Price (USD)")

sns.scatterplot(data=df.sample(4000, random_state=RANDOM_STATE), x="carat", y="price",
                hue="cut", ax=axes[0, 1], alpha=0.5, s=15, palette="viridis")
axes[0, 1].set_title("Carat vs. Price (colored by Cut)")

sns.boxplot(data=df, x="clarity", y="price", order=clarity_order, ax=axes[1, 0])
axes[1, 0].set_title("Price by Clarity Grade")
axes[1, 0].tick_params(axis="x", rotation=45)

corr = df[["carat", "depth", "table", "volume", "cut_encoded", "color_encoded",
           "clarity_encoded", "price"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], cbar=False)
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("../outputs/eda_overview.png", bbox_inches="tight")
plt.show()
print("Saved eda_overview.png")

features = ["carat", "volume", "cut_encoded", "color_encoded", "clarity_encoded", "price"]
X = StandardScaler().fit_transform(df[features])

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=5000, replace=False)

inertias, sil_scores = [], []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X[sample_idx], labels[sample_idx]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), sil_scores, marker="o", color="#DD8452")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig("../outputs/cluster_selection.png", bbox_inches="tight")
plt.show()

stat_best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\nSilhouette scores: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
print(f"Purely statistical optimum: k = {stat_best_k}")

best_k = 4
print(f"Business-informed choice: k = {best_k} "
      f"(silhouette={sil_scores[list(k_range).index(best_k)]:.3f}) - "
      "trades a little statistical tightness for actionable granularity.")


final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["segment"] = final_km.fit_predict(X)
segment_arr = df["segment"].values
final_sil = silhouette_score(X[sample_idx], segment_arr[sample_idx])
print(f"Final model: K-Means with k={best_k}, silhouette score = {final_sil:.3f} (evaluated on 5,000-point sample)")

segment_profile = df.groupby("segment")[["carat", "price", "cut_encoded",
                                          "color_encoded", "clarity_encoded"]].mean().round(2)
segment_profile["count"] = df.groupby("segment").size()
segment_profile["pct_of_catalogue"] = (segment_profile["count"] / len(df) * 100).round(1)
print("\nSegment profiles:\n", segment_profile)


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df.sample(5000, random_state=RANDOM_STATE), x="carat", y="price",
                 hue="segment", palette="Set2", alpha=0.6, s=18)
plt.title(f"Diamond Segments Found by K-Means (k={best_k})")
plt.savefig("../outputs/cluster_scatter.png", bbox_inches="tight")
plt.show()
print("Saved cluster_scatter.png")

df.to_csv("../outputs/diamonds_with_segments.csv", index=False)
print("Saved enriched dataset with segment labels to outputs/diamonds_with_segments.csv")
print("\nProject 1 complete.")