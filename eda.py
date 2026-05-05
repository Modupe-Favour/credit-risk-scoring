import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_data(path):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df

def default_distribution(df):
    counts = df["loan_status"].value_counts().reset_index()
    counts.columns = ["loan_status", "Count"]
    counts["loan_status"] = counts["loan_status"].map(
        {1: "Default", 0: "No Default"})
    fig = px.pie(counts, names="loan_status", values="Count",
                 title="Loan Default Distribution",
                 color_discrete_sequence=["#EF553B", "#00CC96"])
    return fig

def default_by_intent(df):
    data = df.groupby("loan_intent")["loan_status"].mean().reset_index()
    data["loan_status"] = (data["loan_status"] * 100).round(2)
    data = data.sort_values("loan_status", ascending=False)
    fig = px.bar(data, x="loan_intent", y="loan_status",
                 title="Default Rate by Loan Intent (%)",
                 color="loan_status", color_continuous_scale="Reds",
                 labels={"loan_status": "Default Rate (%)"})
    return fig

def default_by_age(df):
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["person_age"],
                             bins=[18, 25, 35, 45, 55, 70],
                             labels=["18-25", "25-35", "35-45",
                                     "45-55", "55-70"])
    data = df.groupby("AgeGroup")["loan_status"].mean().reset_index()
    data["loan_status"] = (data["loan_status"] * 100).round(2)
    fig = px.bar(data, x="AgeGroup", y="loan_status",
                 title="Default Rate by Age Group (%)",
                 color="loan_status", color_continuous_scale="Oranges",
                 labels={"loan_status": "Default Rate (%)"})
    return fig

def default_by_loan_amount(df):
    df = df.copy()
    df["DefaultLabel"] = df["loan_status"].map(
        {1: "Default", 0: "No Default"})
    fig = px.histogram(df, x="loan_amnt", color="DefaultLabel",
                       barmode="overlay",
                       title="Loan Amount by Default Status",
                       color_discrete_sequence=["#EF553B", "#00CC96"])
    return fig

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    top_corr = numeric_df.corr()["loan_status"].abs().sort_values(
        ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df[top_corr].corr(), annot=True,
                fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig