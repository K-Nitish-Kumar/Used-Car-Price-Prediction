import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/car_details.csv")

# Normalize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Support common Kaggle column variations
df = df.rename(columns={
    "car_name": "name",
    "kms_driven": "km_driven",
    "fuel_type": "fuel"
})

# Selling price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], kde=True)
plt.title("Distribution of Used Car Prices")
plt.xlabel("Selling Price")
plt.savefig("outputs/price_distribution.png")
plt.show()

# Car age vs selling price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="year",
    y="selling_price"
)
plt.title("Manufacturing Year vs Selling Price")
plt.savefig("outputs/year_vs_price.png")
plt.show()

# Correlation between numeric features
plt.figure(figsize=(8, 5))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.savefig("outputs/correlation.png")
plt.show()

print("Visualizations generated successfully!")
