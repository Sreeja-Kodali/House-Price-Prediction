import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

spec = importlib.util.spec_from_file_location("houseprice_module", "HousePricePrediction.py")
houseprice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(houseprice)

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stAppViewContainer"] > div:first-child {padding-top: 1rem;}
    .stDataFrame td, .stDataFrame th {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dashboard Navigation")
section = st.sidebar.radio("Go to section:", ["Dataset Overview", "Data Cleaning", "Visualizations"])

try:
    df = pd.read_excel("House Price Prediction Dataset.xlsx")
except Exception as e:
    st.error("❌ Could not load the dataset. Please check the file name/path.")
    st.stop()

if section == "Dataset Overview":
    st.title("🏠 House Price Prediction")
    st.header("📋 Dataset Overview")

    st.write("**Shape of the dataset:**", df.shape)
    st.write("**Data Types:**")
    st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), use_container_width=True)

    st.write("**Summary Statistics:**")
    st.dataframe(df.describe().T.reset_index(), use_container_width=True)

elif section == "Data Cleaning":
    st.header(" Data Cleaning")

    # Missing Values Before & After
    before_missing = df.isnull().sum().reset_index()
    before_missing.columns = ['Column', 'Missing Values Before']

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    after_missing = df.isnull().sum().reset_index()
    after_missing.columns = ['Column', 'Missing Values After']

    missing_df = pd.merge(before_missing, after_missing, on='Column')
    st.subheader("📉 Missing Values (Before vs After Cleaning)")
    st.dataframe(missing_df, use_container_width=True)

    # Outliers Before & After
    def count_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return ((data[col] < lower) | (data[col] > upper)).sum()

    outlier_data = []
    for col in num_cols:
        before = count_outliers(df, col)
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
        after = count_outliers(df, col)
        outlier_data.append([col, before, after])

    outlier_df = pd.DataFrame(outlier_data, columns=['Column', 'Before Capping', 'After Capping'])
    st.subheader("📊 Outliers Before vs After Capping")
    st.dataframe(outlier_df, use_container_width=True)

    st.success("✅ Data cleaning completed successfully!")

    # Download Button
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="📂 Download Cleaned Dataset",
        data=buffer,
        file_name="cleaned_house_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif section == "Visualizations":
    st.header("📊 Visualizations")
    sns.set(style="whitegrid")

    # Sale Price Distribution
    if 'SalePrice' in df.columns:
        st.subheader("Sale Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['SalePrice'], kde=True, color='skyblue', ax=ax)
        ax.set_title("Distribution of Sale Prices", fontsize=14, fontweight='bold')
        st.pyplot(fig)

    # Correlation Heatmap (Top 15)
    st.subheader("Correlation Heatmap (Top 15 Most Correlated with Sale Price)")
    corr = df.corr(numeric_only=True)

    if 'SalePrice' in corr.columns:
        top_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(15).index
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of Top 15 Features", fontsize=14, fontweight='bold')
        st.pyplot(fig)
    else:
        st.warning("'SalePrice' column not found in the dataset.")

    # Average Sale Price by Neighborhood
    if 'Neighborhood' in df.columns:
        st.subheader("Top 10 Neighborhoods by Average Sale Price")
        fig, ax = plt.subplots(figsize=(12, 6))
        neigh_avg = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=neigh_avg.index, y=neigh_avg.values, ax=ax, color='teal')
        ax.set_title("Top 10 Neighborhoods by Average Sale Price")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    # Average Sale Price Over the Years
    if 'YearBuilt' in df.columns and 'SalePrice' in df.columns:
        st.subheader("Average Sale Price Over the Years")
        year_avg = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=year_avg, x='YearBuilt', y='SalePrice', marker='o', color='teal', ax=ax)
        ax.set_title("Average Sale Price by Year Built")
        ax.set_xlabel("Year Built")
        ax.set_ylabel("Average Sale Price")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    # Sale Price vs GrLivArea
    if 'GrLivArea' in df.columns:
        st.subheader("Sale Price vs Living Area (GrLivArea)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', color='orange', ax=ax)
        ax.set_title("Relationship Between Living Area and Sale Price")
        st.pyplot(fig)

    # Sale Price vs Overall Quality
    if 'OverallQual' in df.columns:
        st.subheader(" Sale Price vs Overall Quality")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x='OverallQual', y='SalePrice', ax=ax, palette='Set2')
        ax.set_title("Sale Price Distribution Across Overall Quality Levels")
        st.pyplot(fig)

    st.success("✅ All visualizations loaded successfully!")
