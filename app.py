import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.loader import load_csv, load_demo_data
from src.preprocess import basic_cleaning, dataset_quality_report, missing_values_table
from src.analysis import (
    summary_statistics,
    correlation_matrix,
    grouped_summary,
    time_trend,
    top_correlations,
)
from src.insights import explain_dataset, generate_insights
from src.utils import get_numeric_columns, get_categorical_columns, get_datetime_columns


st.set_page_config(page_title="TrendLens", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            padding:16px 18px;
            border:1px solid #e8e8e8;
            border-radius:14px;
            background:#fafafa;
            height:100%;
        ">
            <div style="font-size:0.9rem;color:#666;margin-bottom:6px;">{label}</div>
            <div style="font-size:1.7rem;font-weight:600;color:#111;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.title("TrendLens")
st.caption("A clean analytical app for profiling datasets, exploring structure, and generating insight.")

top1, top2 = st.columns([3, 2])

with top1:
    st.write(
        "TrendLens turns uploaded CSV data into structured analysis through dataset profiling, "
        "statistical summaries, visual exploration, clustering, and auto-generated insight reporting."
    )

with top2:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    demo_clicked = st.button("Use Demo Dataset", use_container_width=True)

if demo_clicked:
    df = load_demo_data()
    df = basic_cleaning(df)
    st.session_state["df"] = df

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    df = basic_cleaning(df)
    st.session_state["df"] = df

df = st.session_state.get("df")

if df is None:
    st.info("Upload a CSV or load the demo dataset to begin.")
    st.stop()

quality = dataset_quality_report(df)
numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)
datetime_cols = get_datetime_columns(df)

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Rows", f"{quality['rows']}")
with m2:
    metric_card("Columns", f"{quality['columns']}")
with m3:
    metric_card("Missing Values", f"{quality['missing']}")
with m4:
    metric_card("Duplicates", f"{quality['duplicates']}")

st.markdown("### Dataset Explanation")
for note in explain_dataset(df, numeric_cols, categorical_cols, datetime_cols):
    st.write(f"- {note}")

tabs = st.tabs([
    "Overview",
    "Data Quality",
    "Statistics",
    "Visual Analysis",
    "Clustering",
    "Insights Report",
])

with tabs[0]:
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Numeric Columns**")
        st.write(numeric_cols if numeric_cols else "None")
    with c2:
        st.markdown("**Categorical Columns**")
        st.write(categorical_cols if categorical_cols else "None")
    with c3:
        st.markdown("**Datetime Columns**")
        st.write(datetime_cols if datetime_cols else "None")

    st.subheader("Column Types")
    dtype_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Missing Value Analysis")
    missing_df = missing_values_table(df)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    if not missing_df.empty:
        fig_missing = px.bar(
            missing_df,
            x="column",
            y="missing_percent",
            title="Missing Percentage by Column"
        )
        st.plotly_chart(fig_missing, use_container_width=True)

with tabs[2]:
    st.subheader("Summary Statistics")
    stats_df = summary_statistics(df)
    if stats_df.empty:
        st.info("No numeric columns available for statistical analysis.")
    else:
        st.dataframe(stats_df.round(2), use_container_width=True, hide_index=True)

        chosen_col = st.selectbox("Inspect numeric distribution", numeric_cols)
        fig_hist = px.histogram(df, x=chosen_col, nbins=30, title=f"Distribution of {chosen_col}")
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(df, y=chosen_col, title=f"Box Plot of {chosen_col}")
        st.plotly_chart(fig_box, use_container_width=True)

with tabs[3]:
    st.subheader("Visual Analysis")

    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Scatter Analysis**")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
            fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("At least two numeric columns are needed.")

    with v2:
        st.markdown("**Correlation Analysis**")
        corr_df = correlation_matrix(df)
        if corr_df.empty:
            st.info("At least two numeric columns are needed for correlation analysis.")
        else:
            fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

            top_corr_df = top_correlations(corr_df).head(10)
            st.dataframe(top_corr_df.round(2), use_container_width=True, hide_index=True)

    st.markdown("---")

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Grouped Comparison**")
        if categorical_cols and numeric_cols:
            group_col = st.selectbox("Group by", categorical_cols)
            value_col = st.selectbox("Measure", numeric_cols, key="group_measure")
            grouped_df = grouped_summary(df, group_col, value_col)

            st.dataframe(grouped_df, use_container_width=True, hide_index=True)

            fig_group = px.bar(grouped_df, x=group_col, y="mean", title=f"Average {value_col} by {group_col}")
            st.plotly_chart(fig_group, use_container_width=True)

            fig_box_group = px.box(df, x=group_col, y=value_col, title=f"{value_col} by {group_col}")
            st.plotly_chart(fig_box_group, use_container_width=True)
        else:
            st.info("This dataset needs at least one categorical and one numeric column.")

    with g2:
        st.markdown("**Time Trend**")
        if datetime_cols and numeric_cols:
            date_col = st.selectbox("Date field", datetime_cols)
            trend_col = st.selectbox("Trend variable", numeric_cols, key="trend_var")
            trend_df = time_trend(df, date_col, trend_col)
            fig_trend = px.line(trend_df, x=date_col, y=trend_col, markers=True, title=f"{trend_col} over time")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Time trend analysis needs at least one datetime and one numeric column.")

with tabs[4]:
    st.subheader("Clustering & Segmentation")

    if len(numeric_cols) < 2:
        st.info("At least two numeric columns are needed for clustering.")
    else:
        default_cluster_cols = numeric_cols[: min(3, len(numeric_cols))]
        cluster_cols = st.multiselect(
            "Select numeric columns for clustering",
            numeric_cols,
            default=default_cluster_cols
        )

        if len(cluster_cols) < 2:
            st.warning("Please select at least two numeric columns.")
        else:
            X = df[cluster_cols].dropna()

            if len(X) < 3:
                st.warning("Not enough complete rows available after removing missing values.")
            else:
                k = st.slider("Number of clusters", 2, 6, 3)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)

                plot_df = X.copy()
                plot_df["cluster"] = labels.astype(str)

                st.dataframe(plot_df.head(20), use_container_width=True, hide_index=True)

                fig_cluster = px.scatter(
                    plot_df,
                    x=cluster_cols[0],
                    y=cluster_cols[1],
                    color="cluster",
                    title="Cluster Projection"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)

                st.write(
                    "This view groups similar observations based on their numeric patterns. "
                    "It can support segmentation, anomaly discovery, and exploratory structure detection."
                )

with tabs[5]:
    st.subheader("Generated Insight Report")

    missing_df = missing_values_table(df)
    corr_df = correlation_matrix(df)
    insight_list = generate_insights(df, missing_df, corr_df)

    for item in insight_list:
        st.write(f"- {item}")

    report_text = "\n".join(f"- {item}" for item in insight_list)
    st.download_button(
        "Download Insight Summary",
        data=report_text,
        file_name="trendlens_insights.txt",
        mime="text/plain"
    )