from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DS01 = ROOT / "DS-01_Customer_Lifetime_Value_and_Churn" / "data" / "processed"
DS02 = ROOT / "DS-02_Demand_Forecasting_and_Inventory_Optimization" / "data" / "processed"

st.set_page_config(
    page_title="Nexvia | Data Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,.08);
    border: 1px solid rgba(128,128,128,.18);
    padding: 14px 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading project datasets...")
def load_data():
    files = {
        "clv": DS01 / "customer_clv_segmentation.csv",
        "clusters": DS01 / "customer_cluster_profiles.csv",
        "survival": DS01 / "cox_survival_results.csv",
        "uplift": DS01 / "customer_uplift_results.csv",
        "forecast": DS02 / "forecast_results.csv",
        "aggregate": DS02 / "aggregate_forecast.csv",
        "inventory": DS02 / "inventory_recommendations.csv",
        "category": DS02 / "category_inventory.csv",
        "store": DS02 / "store_inventory.csv",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing dashboard data files:\n" + "\n".join(missing))
    data = {name: pd.read_csv(path) for name, path in files.items()}
    data["forecast"]["date"] = pd.to_datetime(data["forecast"]["date"], errors="coerce")
    data["aggregate"]["date"] = pd.to_datetime(data["aggregate"]["date"], errors="coerce")
    data["aggregate"]["covered"] = (
        data["aggregate"]["covered"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )
    return data


try:
    D = load_data()
except Exception as exc:
    st.error("The dashboard could not load its project datasets.")
    st.exception(exc)
    st.stop()


def money(value):
    return f"${value:,.2f}"


def wape(actual, predicted):
    denominator = pd.Series(actual).abs().sum()
    if denominator == 0:
        return float("nan")
    return pd.Series(actual).sub(pd.Series(predicted)).abs().sum() / denominator * 100


with st.sidebar:
    st.markdown("# NEXVIA")
    st.caption("DATA INTELLIGENCE PLATFORM")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Platform Overview", "Customer Intelligence", "Demand & Inventory"],
    )
    st.divider()
    st.caption("Data Science Internship | 2026")
    st.markdown("**Developed by Saud Jabar Ali**")


# ---------------------------------------------------------------------
# PLATFORM OVERVIEW
# ---------------------------------------------------------------------
if page == "Platform Overview":
    st.title("Nexvia Intelligence Platform")
    st.caption("Customer Intelligence · Demand Forecasting · Inventory Optimization")
    st.write(
        "An integrated analytics platform combining customer lifetime value "
        "and churn intelligence with multi-series demand forecasting and "
        "budget-constrained inventory recommendations."
    )

    clv = D["clv"]
    inv = D["inventory"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer Records", f"{len(clv):,}")
    c2.metric("Product-Store Series", f"{inv['id'].nunique():,}")
    c3.metric("Stores Analysed", f"{D['store']['store_id'].nunique():,}")
    c4.metric("Total Estimated CLV", money(clv["EstimatedCLV"].sum()))

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("👥 Customer Intelligence")
        st.markdown(
            "- Customer churn analysis\n"
            "- Customer lifetime value estimation\n"
            "- K-means customer segmentation\n"
            "- Cox proportional hazards analysis\n"
            "- Exploratory uplift modelling"
        )
    with right:
        st.subheader("📦 Demand & Inventory")
        st.markdown(
            "- M5 retail demand forecasting\n"
            "- Global LightGBM forecasting model\n"
            "- Naive baseline comparison\n"
            "- Aggregate prediction intervals\n"
            "- OR-Tools inventory allocation"
        )

    st.divider()
    st.subheader("Project scope and interpretation")
    st.warning(
        "CLV values are illustrative estimates based on stated retention, "
        "margin and discount assumptions. Uplift treatment assignments are "
        "synthetic and do not establish causal campaign effects."
    )
    st.warning(
        "Demand model evaluation uses a stratified development sample of "
        "600 product-store series, not the complete 30,490-series M5 hierarchy. "
        "Inventory and cost inputs are scenario assumptions."
    )


# ---------------------------------------------------------------------
# CUSTOMER INTELLIGENCE
# ---------------------------------------------------------------------
elif page == "Customer Intelligence":
    clv = D["clv"].copy()
    clusters = D["clusters"].copy()
    survival = D["survival"].copy()
    uplift = D["uplift"].copy()

    st.title("Customer Intelligence Dashboard")
    st.caption("Customer Lifetime Value · Churn · Segmentation · Survival · Uplift")

    total = len(clv)
    churned = int((clv["Churn"].astype(str).str.lower() == "yes").sum())
    churn_rate = churned / total * 100 if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Observed Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Average Estimated CLV", money(clv["EstimatedCLV"].mean()))
    c4.metric("Total Estimated CLV", money(clv["EstimatedCLV"].sum()))

    t_overview, t_clv, t_segments, t_survival, t_uplift = st.tabs(
        ["Overview", "CLV Analysis", "Customer Segments", "Survival Analysis", "Uplift Analysis"]
    )

    with t_overview:
        st.subheader("Observed Customer Churn")
        counts = (
            clv["Churn"].fillna("Unknown").astype(str).value_counts()
            .rename_axis("Churn").reset_index(name="Customers")
        )
        left, right = st.columns(2)
        with left:
            fig = px.pie(counts, names="Churn", values="Customers", hole=.55,
                         title="Customer Churn Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(clv, x="tenure", color="Churn", nbins=30,
                               barmode="overlay", opacity=.75,
                               title="Customer Tenure Distribution",
                               labels={"tenure": "Tenure (months)"})
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            clv, x="tenure", y="MonthlyCharges", color="Churn",
            hover_data=["CLV_Segment", "EstimatedCLV"], opacity=.65,
            title="Monthly Charges vs Customer Tenure",
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with t_clv:
        st.subheader("Customer Lifetime Value by Segment")
        summary = (
            clv.groupby("CLV_Segment", dropna=False)
            .agg(
                Customers=("customerID", "count"),
                Average_CLV=("EstimatedCLV", "mean"),
                Median_CLV=("EstimatedCLV", "median"),
                Total_CLV=("EstimatedCLV", "sum"),
            ).reset_index()
        )
        left, right = st.columns(2)
        with left:
            fig = px.bar(summary, x="CLV_Segment", y="Average_CLV",
                         color="CLV_Segment", title="Average Estimated CLV by Segment")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.pie(summary, names="CLV_Segment", values="Customers",
                         hole=.45, title="Customers by CLV Segment")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "CLV is an illustrative estimate based on the project's assumed "
            "gross margin, retention and discount rate."
        )
        with st.expander("View customer-level CLV records"):
            st.dataframe(
                clv[["customerID", "tenure", "MonthlyCharges", "Churn",
                     "PredictedChurnProbability", "EstimatedCLV", "CLV_Segment"]],
                use_container_width=True, hide_index=True,
            )

    with t_segments:
        st.subheader("K-means Customer Segmentation")
        left, right = st.columns(2)
        with left:
            fig = px.bar(clusters, x="CustomerCluster", y="CustomerCount",
                         color="CustomerCluster", title="Customers per Cluster")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                clusters, x="AvgTenure", y="AvgMonthlyCharges",
                size="CustomerCount", color="CustomerCluster",
                hover_data=["AvgCLV", "AvgChurnProbability", "AvgMonthlyGrossProfit"],
                title="Cluster Characteristics",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(clusters, use_container_width=True, hide_index=True)

    with t_survival:
        st.subheader("Cox Proportional Hazards Analysis")
        sd = survival.sort_values("Hazard_Ratio").copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sd["Hazard_Ratio"], y=sd["covariate"], mode="markers",
            marker={"size": 9},
            error_x={
                "type": "data", "symmetric": False,
                "array": sd["HR_CI_Upper"] - sd["Hazard_Ratio"],
                "arrayminus": sd["Hazard_Ratio"] - sd["HR_CI_Lower"],
            },
            name="Hazard Ratio",
        ))
        fig.add_vline(x=1, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Hazard Ratios with 95% Confidence Intervals",
            xaxis_title="Hazard Ratio", yaxis_title="Covariate", height=650,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Hazard ratios below 1 indicate a lower estimated hazard relative "
            "to the model reference category; values above 1 indicate a higher "
            "estimated hazard. Interpret alongside confidence intervals and p-values."
        )
        st.dataframe(survival, use_container_width=True, hide_index=True)

    with t_uplift:
        st.subheader("Exploratory Customer Uplift Analysis")
        counts = (
            uplift["Uplift_Category"].fillna("Unknown").value_counts()
            .rename_axis("Uplift Category").reset_index(name="Customers")
        )
        left, right = st.columns(2)
        with left:
            fig = px.bar(counts, x="Uplift Category", y="Customers",
                         color="Uplift Category", title="Customers by Uplift Category")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(
                uplift, x="Uplift_Score", color="Uplift_Category", nbins=50,
                title="Uplift Score Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.warning(
            "Treatment assignments in this project are synthetic. These "
            "model-based uplift scores are exploratory and do not establish "
            "causal campaign effectiveness."
        )


# ---------------------------------------------------------------------
# DEMAND FORECASTING & INVENTORY
# ---------------------------------------------------------------------
elif page == "Demand & Inventory":
    forecast = D["forecast"].copy()
    aggregate = D["aggregate"].copy()
    inventory = D["inventory"].copy()
    category = D["category"].copy()
    store = D["store"].copy()

    st.title("Demand Forecasting & Inventory Optimization")
    st.caption("M5 Walmart Dataset · Global LightGBM · Prediction Intervals · OR-Tools")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Product-Store Series", f"{inventory['id'].nunique():,}")
    c2.metric("Stores", f"{store['store_id'].nunique():,}")
    c3.metric("Recommended Order Units", f"{inventory['recommended_order'].sum():,.0f}")
    c4.metric("Recommended Procurement Cost", money(inventory["order_cost"].sum()))

    t_forecast, t_evaluation, t_inventory = st.tabs(
        ["Demand Forecasting", "Forecast Evaluation", "Inventory Optimization"]
    )

    with t_forecast:
        st.subheader("Actual vs Forecast Demand")
        daily = (
            forecast.groupby("date", as_index=False)
            .agg(Actual=("sales", "sum"), Forecast=("forecast", "sum"))
            .sort_values("date")
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["Actual"], mode="lines+markers",
            name="Actual Demand",
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["Forecast"], mode="lines+markers",
            name="Forecast Demand",
        ))
        fig.update_layout(
            title="Daily Aggregate Demand: Actual vs Forecast",
            xaxis_title="Date", yaxis_title="Units Sold", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Aggregate Prediction Interval")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=aggregate["date"], y=aggregate["upper_bound"], mode="lines",
            line={"width": 0}, showlegend=False, name="Upper Bound",
        ))
        fig.add_trace(go.Scatter(
            x=aggregate["date"], y=aggregate["lower_bound"], mode="lines",
            fill="tonexty", fillcolor="rgba(31,119,180,.18)",
            line={"width": 0}, name="Prediction Interval",
        ))
        fig.add_trace(go.Scatter(
            x=aggregate["date"], y=aggregate["forecast"],
            mode="lines+markers", name="Forecast",
        ))
        fig.add_trace(go.Scatter(
            x=aggregate["date"], y=aggregate["actual"],
            mode="lines+markers", name="Actual",
        ))
        fig.update_layout(
            title="Aggregate Demand with Prediction Interval",
            xaxis_title="Date", yaxis_title="Units", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        coverage = aggregate["covered"].mean() * 100 if len(aggregate) else float("nan")
        st.metric("Observed Aggregate Interval Coverage", f"{coverage:.2f}%")
        st.caption(
            "Observed coverage is calculated over the 14-day aggregate "
            "evaluation period. It is an empirical result, not a guarantee "
            "of future coverage."
        )

    with t_evaluation:
        st.subheader("Forecast Model Performance")
        left, right = st.columns(2)
        with left:
            st.metric("Naive Baseline WAPE", "101.11%")
            st.metric("Global LightGBM WAPE", "76.99%")
        with right:
            st.metric("Naive Baseline Mean RMSSE", "1.0495")
            st.metric("Global LightGBM Mean RMSSE", "0.7801")
        st.success(
            "On the evaluated sample, Global LightGBM improved WAPE by "
            "23.86% and mean RMSSE by 25.67% compared with the naive baseline."
        )
        st.metric(
            "WAPE Recalculated from Exported Evaluation Rows",
            f"{wape(forecast['sales'], forecast['forecast']):.2f}%",
        )
        fig = px.histogram(
            forecast, x="error", nbins=60, title="Distribution of Forecast Errors",
            labels={"error": "Forecast Error (actual - forecast)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Evaluation covers 600 sampled product-store series over a 28-day "
            "validation horizon. Mean RMSSE is the unweighted mean across "
            "evaluated series, not official M5 WRMSSE. Aggregate WAPE and "
            "series-level WAPE measure different things."
        )
        with st.expander("View exported forecast evaluation records"):
            st.dataframe(forecast, use_container_width=True, hide_index=True)

    with t_inventory:
        st.subheader("Budget-Constrained Inventory Recommendations")
        current_stock = inventory["current_stock"].sum()
        recommended = inventory["recommended_order"].sum()
        unmet = inventory["unmet_demand"].sum()
        cost = inventory["order_cost"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Stock", f"{current_stock:,.0f}")
        c2.metric("Recommended Order", f"{recommended:,.0f}")
        c3.metric("Unmet Conservative Target", f"{unmet:,.0f}")
        c4.metric("Procurement Cost", money(cost))

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                category, x="Category", y="Order_Cost", color="Category",
                title="Procurement Cost by Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                store.sort_values("Order_Cost", ascending=True),
                x="Order_Cost", y="store_id", orientation="h",
                title="Procurement Cost by Store",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Allocation")
        st.dataframe(category, use_container_width=True, hide_index=True)
        st.subheader("Store Allocation")
        st.dataframe(store, use_container_width=True, hide_index=True)

        st.subheader("Top Replenishment Recommendations")
        top = inventory.sort_values("recommended_order", ascending=False).head(20)
        columns = [
            "item_id", "store_id", "cat_id", "Forecast_Demand",
            "current_stock", "recommended_order", "unmet_demand", "order_cost",
        ]
        st.dataframe(top[columns], use_container_width=True, hide_index=True)

        st.warning(
            "Current stock, unit costs and budget are simulated scenario "
            "assumptions. 'Unmet conservative target' is measured against the "
            "project's target-stock calculation; it is not an observed "
            "real-world stockout measure."
        )
        st.download_button(
            "Download Inventory Recommendations (CSV)",
            data=inventory.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
        )


st.divider()
st.markdown(
    """
    <div style="text-align:center; padding: 12px 0 4px 0;">
        <div style="font-size: 0.95rem; font-weight: 600;">
            Developed by Saud Jabar Ali
        </div>
        <div style="font-size: 0.82rem; color: #8b949e; margin-top: 4px;">
            Nexvia Technologies | Data Science Internship | 2026
        </div>
        <div style="font-size: 0.78rem; color: #8b949e;">
            Customer Intelligence &amp; Demand Forecasting Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
