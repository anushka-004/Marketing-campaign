import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Marketing Campaign Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
sns.set_style("whitegrid")

# ============================================================
# 2. DATA LOADING + CLEANING (same steps as the notebook)
# ============================================================
@st.cache_data
def load_data(path="Data/marketing_campaign_data.csv"):
    df = pd.read_csv(path)

    # Dt_Customer -> datetime
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])

    # Fix messy Marital_Status categories
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"YOLO": "Other", "Absurd": "Other", "Alone": "Single"}
    )

    current_year = datetime.now().year

    # Feature engineering (identical to notebook)
    df["Age"] = current_year - df["Year_Birth"]
    df["Customer_Tenure"] = current_year - df["Dt_Customer"].dt.year
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["Total_Spend"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Purchases"] = (
        df["NumDealsPurchases"] + df["NumWebPurchases"]
        + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )

    # Rule based segmentation (from project brief)
    spend_90th = df["Total_Spend"].quantile(0.90)
    df["Seg_High_Income"] = df["Income"] > 75000
    df["Seg_Young_Customer"] = df["Age"] < 30
    df["Seg_Campaign_Responder"] = df["Response"] == 1
    df["Seg_High_Web_Engagement"] = df["NumWebVisitsMonth"] > 5
    df["Seg_Family_Customer"] = df["Children"] > 0
    df["Seg_High_Spender"] = df["Total_Spend"] > spend_90th

    # Age band / Income band for filters (required by project guidelines)
    df["Age_Band"] = pd.cut(
        df["Age"], bins=[0, 30, 40, 50, 60, 70, 120],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    )
    df["Income_Band"] = pd.cut(
        df["Income"], bins=[0, 25000, 50000, 75000, 100000, np.inf],
        labels=["<25k", "25-50k", "50-75k", "75-100k", "100k+"]
    )

    return df

try:
    marketing_df = load_data()
except FileNotFoundError:
    st.error(
        "Couldn't find **marketing_campaign_data.csv**. "
        "Place it in the same folder as this app, or upload it below."
    )
    uploaded = st.file_uploader("Upload marketing_campaign_data.csv", type="csv")
    if uploaded is not None:
        marketing_df = load_data(uploaded)
    else:
        st.stop()

# ============================================================
# 3. SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔎 Filters")

countries = st.sidebar.multiselect(
    "Country", options=sorted(marketing_df["Country"].unique()),
    default=sorted(marketing_df["Country"].unique())
)
education = st.sidebar.multiselect(
    "Education", options=sorted(marketing_df["Education"].unique()),
    default=sorted(marketing_df["Education"].unique())
)
marital = st.sidebar.multiselect(
    "Marital Status", options=sorted(marketing_df["Marital_Status"].unique()),
    default=sorted(marketing_df["Marital_Status"].unique())
)
age_range = st.sidebar.slider(
    "Age range",
    int(marketing_df["Age"].min()), int(marketing_df["Age"].max()),
    (int(marketing_df["Age"].min()), int(marketing_df["Age"].max()))
)
income_range = st.sidebar.slider(
    "Income range (₹)",
    int(marketing_df["Income"].min()), int(marketing_df["Income"].max()),
    (int(marketing_df["Income"].min()), int(marketing_df["Income"].max()))
)

segment_filter = st.sidebar.selectbox(
    "Quick segment view",
    ["All Customers", "High Income", "Young Customer", "Campaign Responder",
     "High Web Engagement", "Family Customer", "High Spender"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data cleaned & feature-engineered exactly as in the EDA notebook.")

# Apply filters
df = marketing_df[
    marketing_df["Country"].isin(countries)
    & marketing_df["Education"].isin(education)
    & marketing_df["Marital_Status"].isin(marital)
    & marketing_df["Age"].between(*age_range)
    & marketing_df["Income"].between(*income_range)
].copy()

segment_map = {
    "High Income": "Seg_High_Income",
    "Young Customer": "Seg_Young_Customer",
    "Campaign Responder": "Seg_Campaign_Responder",
    "High Web Engagement": "Seg_High_Web_Engagement",
    "Family Customer": "Seg_Family_Customer",
    "High Spender": "Seg_High_Spender",
}
if segment_filter != "All Customers":
    df = df[df[segment_map[segment_filter]]]

if df.empty:
    st.warning("No customers match the current filters. Try widening them.")
    st.stop()

# ============================================================
# 4. HEADER + KPI CARDS
# ============================================================
st.title("📊 Marketing Campaign Analytics Dashboard")
st.caption("Interactive view of customer demographics, spending, and campaign response.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Response Rate", f"{df['Response'].mean()*100:.1f}%")
k3.metric("Avg Income", f"₹{df['Income'].mean():,.0f}")
k4.metric("Avg Total Spend", f"₹{df['Total_Spend'].mean():,.0f}")
k5.metric("Avg Purchases", f"{df['Total_Purchases'].mean():.1f}")

st.markdown("---")

# ============================================================
# Helper to render a matplotlib/seaborn figure in Streamlit
# ============================================================
def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ============================================================
# 5. TABS FOR EACH TYPE OF ANALYSIS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Univariate", "🔗 Bivariate", "🧬 Multivariate & Correlation", "🎯 Segmentation"]
)

# ---- TAB 1: Univariate (Income, Age, Total_Spend, Recency) ----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Income"], bins=50, ax=ax)
        ax.set_title("Distribution of Customer Income")
        ax.set_xlabel("Income")
        ax.set_ylabel("Number of Customers")
        show_fig(fig)
        st.caption("Income is positively skewed — most customers sit in the low-to-middle range, "
                   "with high-income customers being relatively rare.")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Total_Spend"], bins=30, ax=ax)
        ax.set_title("Distribution of Total Expenditure")
        ax.set_xlabel("Total_Spend")
        ax.set_ylabel("Number of Customers")
        show_fig(fig)
        st.caption("Most customers spend relatively little; a small group of high spenders stands out.")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Age"], bins=30, ax=ax)
        ax.set_title("Age Distribution of Customers")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Customers")
        show_fig(fig)
        st.caption("Customers aged 50–60 form the largest group — a good core segment to target.")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Recency"], bins=30, ax=ax)
        ax.set_title("Distribution of Recency")
        ax.set_xlabel("Recency")
        ax.set_ylabel("Number of Customers")
        show_fig(fig)
        st.caption("Many customers haven't purchased recently — a strong candidate group for "
                   "re-engagement campaigns.")

# ---- TAB 2: Bivariate (Income vs Spend, Response vs Income/Age/Spend) ----
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="Income", y="Total_Spend", alpha=0.3, s=20, ax=ax)
        ax.set_title("Income vs Total Spend")
        show_fig(fig)
        st.caption("Higher income generally means higher spend, with some variation.")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Response", y="Income", ax=ax)
        ax.set_title("Response vs Income")
        show_fig(fig)
        st.caption("Customers who responded to campaigns tend to have higher incomes.")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Response", y="Age", ax=ax)
        ax.set_title("Response vs Age")
        show_fig(fig)
        st.caption("Age shows a smaller difference between responders and non-responders.")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Response", y="Total_Spend", ax=ax)
        ax.set_title("Response vs Total Spend")
        show_fig(fig)
        st.caption("Responders spend noticeably more, on average, than non-responders.")

# ---- TAB 3: Multivariate + Correlation ----
with tab3:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="Income", y="Total_Spend", hue="Response", ax=ax)
    ax.set_title("Income vs Total Spend, colored by Campaign Response")
    show_fig(fig)
    st.caption("High-income, high-spending customers are the most likely to respond — "
               "prime targets for premium/loyalty offers.")

    fig, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(
        df.corr(numeric_only=True), annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    show_fig(fig)
    st.caption("Income, purchase frequency, and spending are the strongest indicators of "
               "customer value — target campaigns using a combination of these factors.")

# ---- TAB 4: Rule-based Segmentation ----
with tab4:
    st.subheader("Segment sizes & response rate")
    seg_rows = []
    for label, col in segment_map.items():
        seg_df = marketing_df[marketing_df[col]]
        seg_rows.append({
            "Segment": label,
            "Customers": len(seg_df),
            "% of Base": f"{len(seg_df)/len(marketing_df)*100:.1f}%",
            "Response Rate": f"{seg_df['Response'].mean()*100:.1f}%",
            "Avg Total Spend": f"₹{seg_df['Total_Spend'].mean():,.0f}",
        })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        by_country = df.groupby("Country", as_index=False).agg(
            Response_Rate=("Response", "mean"), Customers=("ID", "count")
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=by_country, x="Country", y="Response_Rate", ax=ax)
        ax.set_title("Response Rate by Country")
        plt.xticks(rotation=30)
        show_fig(fig)
    with c2:
        by_education = df.groupby("Education", as_index=False).agg(
            Response_Rate=("Response", "mean")
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=by_education, x="Education", y="Response_Rate", ax=ax)
        ax.set_title("Response Rate by Education")
        plt.xticks(rotation=30)
        show_fig(fig)

st.markdown("---")
with st.expander("📋 View filtered raw data"):
    st.dataframe(df, use_container_width=True)