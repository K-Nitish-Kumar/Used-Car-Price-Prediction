import pandas as pd

# Load dataset
df = pd.read_csv("dataset/car_details.csv")

# Display first five records
print("\nFirst five records:")
print(df.head())

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical summary:")
print(df.describe(include="all"))

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())
