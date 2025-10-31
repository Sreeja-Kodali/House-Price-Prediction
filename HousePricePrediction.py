import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

print("🏠 House Price Prediction\n")

# Load Dataset
df = pd.read_excel("House Price Prediction Dataset.xlsx")
print("Original Shape:", df.shape)

# Dataset Overview
print("\n=== Dataset Overview ===")
print("\n📋 Data Types:")
print(df.dtypes)
print("\n📈 Summary Statistics:")
print(df.describe())

# Check Missing Values
print("\n=== Missing Values Before Cleaning ===")
print(df.isnull().sum())

# Separate Numeric & Categorical Columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Handle Missing Values
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# Outlier Detection & Capping
def cap_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col} → Outliers Detected: {len(outliers)}")
    df[col] = df[col].clip(lower, upper)

print("\n=== Outlier Detection ===")
for col in num_cols:
    cap_outliers(col)

print("\n✅ Outliers handled and missing values cleaned successfully!\n")

# Save Cleaned Data
df.to_csv("cleaned_house_data.csv", index=False)
print("Cleaned data saved to 'cleaned_house_data.csv'")

sns.set(style="whitegrid")

# Sale Price Distribution
if 'SalePrice' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['SalePrice'], kde=True, color='skyblue')
    plt.title("Distribution of House Sale Prices", fontsize=14, fontweight='bold')
    plt.xlabel("Sale Price")
    plt.ylabel("Frequency")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)

if 'SalePrice' in corr.columns:
    top_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(15).index
    sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title("Correlation Heatmap of Top 15 Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 'SalePrice' column not found in the dataset.")


# Average Sale Price by Neighborhood
if 'Neighborhood' in df.columns:
    plt.figure(figsize=(12,6))
    neigh_avg = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=neigh_avg.index, y=neigh_avg.values)
    plt.title("Top 10 Neighborhoods by Average Sale Price", fontsize=14, fontweight='bold')
    plt.xlabel("Neighborhood")
    plt.ylabel("Average Sale Price")
    plt.xticks(rotation=45)
    plt.show()


# Average Sale Price Over the Years
if 'YearBuilt' in df.columns and 'SalePrice' in df.columns:
    year_avg = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(data=year_avg, x='YearBuilt', y='SalePrice', marker='o', color='teal')
    plt.title("Trend of Average Sale Price Over the Years", fontsize=14, fontweight='bold')
    plt.xlabel("Year Built", fontsize=12)
    plt.ylabel("Average Sale Price", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Sale Price vs Living Area
if 'GrLivArea' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', color='orange')
    plt.title("Sale Price vs Living Area (GrLivArea)", fontsize=14, fontweight='bold')
    plt.xlabel("GrLivArea (sq ft)")
    plt.ylabel("Sale Price")
    plt.show()


# Sale Price Distribution by Overall Quality
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='OverallQual', y='SalePrice', color='skyblue')
plt.title("Sale Price Distribution by Overall Quality", fontsize=14, fontweight='bold')
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.show()

print("\n✅ Visualization and analysis completed successfully!")
