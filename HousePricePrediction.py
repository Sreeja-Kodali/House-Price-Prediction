import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 🏠 Clean layout without Streamlit default icon
st.set_page_config(page_title="House Price Analysis", page_icon="🏠", layout="wide")

# Hide Streamlit default menu, footer, and header
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    .stApp {background-color: #0E1117;}
    .block-container {max-width: 900px; margin: auto;}
    .dataframe {text-align: center;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 🏠 Title
st.title("🏠 House Price Prediction")

# 1️⃣ Load Dataset
st.header("1️⃣ Load Dataset")
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success(f"✅ Dataset Loaded Successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
    st.dataframe(df.head(), use_container_width=True)

    # 2️⃣ Dataset Overview
    st.header("2️⃣ Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Data Types:**")
    st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Dtype"}), use_container_width=True)
    st.write("**Statistical Summary:**")
    st.dataframe(df.describe().T, use_container_width=True)

    # 3️⃣ Handling Missing Values
    st.header("3️⃣ Handling Missing Values")
    st.markdown("### Missing Values Before Cleaning:")

    missing_before = df.isnull().sum()
    missing_before = missing_before[missing_before > 0].sort_values(ascending=False)
    if not missing_before.empty:
        df_missing = missing_before.reset_index()
        df_missing.columns = ["Column", "Missing Values"]
        st.markdown(
            "<div style='display: flex; justify-content: center;'>",
            unsafe_allow_html=True
        )
        st.dataframe(
            df_missing.style.set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]}
            ]),
            use_container_width=False, width=500
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("✅ No missing values found!")

    # Fill missing values (median/mode)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    st.write("### Missing Values After Cleaning:")
    if df.isnull().sum().sum() == 0:
        st.success("✅ All missing values handled successfully!")

    # 4️⃣ Outlier Detection & Capping
    st.header("4️⃣ Outlier Detection & Capping")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outlier_summary = []

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        count_before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        count_after = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary.append([col, count_before, count_after])

    st.success("✅ Outliers capped successfully using IQR method.")
    st.markdown("### Outlier Summary:")
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(outlier_summary, columns=["Column", "Outliers Before", "Outliers After"]),
                 use_container_width=False, width=500)
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Cleaned dataset is now ready for visualization and analysis.")

    # 🧾 Download Cleaned Data (Before Visuals)
    st.header("📥 Download Cleaned Dataset")
    cleaned_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Cleaned Data as CSV",
        data=cleaned_csv,
        file_name="cleaned_house_data.csv",
        mime="text/csv"
    )

    # 5️⃣ Visualizations & Insights
    st.header("5️⃣ Visualizations & Insights")
    st.markdown("### 📊 Visual Analysis of Cleaned Data")
    st.write("Below are compact, easy-to-read charts for understanding data trends.")

    sns.set(style="whitegrid")

    def small_plot(fig):
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        plt.close(fig)

    # 🏠 1. Sale Price Distribution
    if 'SalePrice' in df.columns:
        st.subheader("🏠 Sale Price Distribution")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.histplot(df['SalePrice'], kde=True, ax=ax, color='skyblue')
        ax.set_title("Distribution of Sale Price")
        small_plot(fig)

    # 🌆 2. Neighborhood vs Sale Price
    if 'Neighborhood' in df.columns and 'SalePrice' in df.columns:
        st.subheader("🌆 Top 10 Neighborhoods by Average Sale Price")
        fig, ax = plt.subplots(figsize=(5,3))
        top_neigh = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=top_neigh.index, y=top_neigh.values, ax=ax, palette='crest')
        ax.set_title("Top 10 Neighborhoods by Avg Sale Price")
        plt.xticks(rotation=45)
        small_plot(fig)

    # 📐 3. Sale Price vs Living Area
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        st.subheader("📐 Sale Price vs Living Area")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], ax=ax, color='tomato')
        ax.set_title("Sale Price vs Living Area")
        small_plot(fig)

    # 💎 4. Sale Price vs Overall Quality
    if 'OverallQual' in df.columns and 'SalePrice' in df.columns:
        st.subheader("💎 Sale Price vs Overall Quality")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.boxplot(x=df['OverallQual'], y=df['SalePrice'], ax=ax, palette='Set2')
        ax.set_title("Sale Price vs Overall Quality")
        small_plot(fig)

    # 🔥 5. Correlation Heatmap
    numeric_df = df.select_dtypes(include=['int64','float64'])
    if not numeric_df.empty:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        small_plot(fig)

    st.success("✅ Visualization and Analysis Completed Successfully!")

else:
    st.info("👆 Please upload your dataset file to begin.")
