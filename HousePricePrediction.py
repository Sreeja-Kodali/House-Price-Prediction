import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)
st.title("Housing Price Prediction ")

# Load Dataset
df = pd.read_excel("House Price Prediction Dataset.xlsx")
print("Original Shape:", df.shape)

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
    print(f"\n{col} → Outliers detected: {len(outliers)}")

# Cap outliers
    df[col] = df[col].apply(lambda x: upper if x > upper else (lower if x < lower else x))

# Apply to all numeric columns
for col in num_cols:
    cap_outliers(col)

#  Save Cleaned Data
df.to_csv("cleaned_house_data.csv", index=False)
print("\n✅ Cleaned data with capped outliers saved to 'cleaned_house_data.csv'")

df = pd.read_csv("cleaned_house_data.csv")

sns.set(style="whitegrid")

# Sale Price Distribution
if 'SalePrice' in df.columns:
    st.write("#### Sale Price Distribution")
    plt.figure(figsize=(8,5))
    sns.histplot(df['SalePrice'], kde=True, color='skyblue')
    plt.title("Distribution of House Sale Prices")
    plt.xlabel("Sale Price")
    plt.ylabel("Frequency")
    st.pyplot(plt)


# Correlation Heatmap (numeric columns)
st.write("#### Correlation Heatmap")
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
st.pyplot(plt)


#  Average Sale Price by Neighborhood
if 'Neighborhood' in df.columns:
    st.write("#### Top 10 Neighborhoods by Average Sale Price")
    plt.figure(figsize=(12,6))
    neigh_avg = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=neigh_avg.index, y=neigh_avg.values)
    plt.title("Top 10 Neighborhoods by Average Sale Price")
    plt.xlabel("Neighborhood")
    plt.ylabel("Average Sale Price")
    plt.xticks(rotation=45)
    st.pyplot(plt)


# Sale Price vs GrLivArea (scatter)
if 'GrLivArea' in df.columns:
    st.write("#### Sale Price vs GrLivArea")
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', color='orange')
    plt.title("Sale Price vs Living Area (GrLivArea)")
    plt.xlabel("GrLivArea (sq ft)")
    plt.ylabel("Sale Price")
    st.pyplot(plt)

# Sale Price vs Overall Quality (boxplot)
if 'OverallQual' in df.columns:
    st.write("#### Sale Price vs Overall Quality")
    plt.figure(figsize=(7,4))
    sns.boxplot(data=df, x='OverallQual', y='SalePrice')
    plt.title("Sale Price Distribution by Overall Quality")
    plt.xlabel("Overall Quality")
    plt.ylabel("Sale Price")
    st.pyplot(plt)

print("\n Visualization and analysis completed successfully!")