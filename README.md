# Nexvia Technologies --- Data Science Internship

An integrated **Data Intelligence Platform** built during my Data
Science Internship at Nexvia Technologies. It combines customer
analytics, churn intelligence, demand forecasting, and inventory
optimization in one interactive Streamlit application.

## Live Dashboard

**[Open the Nexvia Data Intelligence
Platform](https://nexvia-data-intelligence-saudjabarali.streamlit.app/)**

The application has three sections: - **Platform Overview** --- overview
of both project workstreams. - **Customer Intelligence** --- CLV, churn,
customer segmentation, survival analysis, and exploratory uplift
analysis. - **Demand & Inventory** --- demand forecasts, evaluation
metrics, prediction intervals, and inventory recommendations.

------------------------------------------------------------------------

## DS-01 --- Customer Lifetime Value and Churn Intelligence

**Objective:** Understand customer churn, estimate customer value,
identify customer groups, and explore retention opportunities.

**Key components** - Exploratory data analysis and feature
engineering. - LightGBM churn classification and SHAP explainability. -
Customer lifetime value (CLV) estimation and segmentation. - K-Means
customer clustering. - Kaplan--Meier survival analysis and Cox
proportional hazards regression, including diagnostics. - PostgreSQL
integration for customer data storage and querying. - Exploratory uplift
modelling using a T-Learner approach with LightGBM.

**Dashboard outputs** - Customer-level CLV and churn-related
estimates. - Customer cluster profiles. - Cox regression hazard-ratio
results. - Uplift scores and uplift categories.

**Interpretation notes** - CLV is estimated using modelling assumptions;
it is not observed future revenue or realized profit. - Churn
predictions are model estimates, not observed month-by-month retention
histories. - The uplift workflow uses synthetic treatment assignment
because genuine campaign-treatment records are unavailable. Its scores
are exploratory, **not causal evidence** or proof of campaign impact.

## DS-02 --- Demand Forecasting and Inventory Optimization

**Objective:** Build a demand forecasting and inventory decision-support
workflow using the M5 forecasting dataset.

**Key components** - Time-series preparation and feature engineering. -
Global LightGBM model with recursive 28-day forecasting. - Comparison
against a naive forecasting baseline. - Aggregate forecast analysis and
prediction intervals. - Bottom-up aggregation of product-store
forecasts. - Inventory recommendations using an OR-Tools CP-SAT
optimization model.

**Dashboard outputs** - Product-store forecast results. - Aggregate
actual-versus-forecast trends. - Forecast error metrics and interval
coverage. - Inventory recommendations by product, category, and store. -
Downloadable inventory recommendation data.

**Reported evaluation results** - Product-store development/evaluation
sample: **600 series** selected from the larger M5 dataset. -
Product-store WAPE: **76.99%** for LightGBM versus **101.11%** for the
naive baseline. - Mean RMSSE: **0.7801** for LightGBM versus **1.0495**
for the naive baseline. - Daily aggregate WAPE: **6.79%**. This
aggregate-level metric is distinct from product-store WAPE. - Aggregate
prediction-interval coverage: **78.57%** over the 14-day evaluation
window.

**Interpretation notes** - The 600 product-store series are a
development/evaluation sample, not the full M5 dataset. - Inventory
optimization uses simulated stock levels, costs, and budget
assumptions---not verified live retailer inventory or procurement
data. - "Unmet target demand" is a modelled gap against the optimization
target; it is not evidence of actual historical stockouts. -
Prediction-interval coverage is measured over a limited 14-day
evaluation window and does not guarantee future coverage.

------------------------------------------------------------------------

## Technology Stack

  -----------------------------------------------------------------------
  Area                                Tools and methods
  ----------------------------------- -----------------------------------
  Programming                         Python

  Data analysis                       Pandas, NumPy

  Machine learning                    LightGBM, Scikit-learn

  Explainability                      SHAP

  Customer analytics                  CLV estimation, K-Means,
                                      Kaplan--Meier, Cox regression

  Uplift exploration                  T-Learner approach

  Forecasting                         LightGBM, recursive forecasting,
                                      naive baseline

  Optimization                        OR-Tools CP-SAT

  Database                            PostgreSQL

  Visualization                       Plotly

  Interactive application             Streamlit
  -----------------------------------------------------------------------

## Repository Structure

``` text
Nexvia-Technologies-Data-Science-Internship/
│
├── app.py
├── requirements.txt
├── README.md
│
├── DS-01_Customer_Lifetime_Value_and_Churn/
│   ├── main.ipynb
│   └── data/
│       └── processed/
│           ├── cox_survival_results.csv
│           ├── customer_cluster_profiles.csv
│           ├── customer_clv_segmentation.csv
│           └── customer_uplift_results.csv
│
└── DS-02_Demand_Forecasting_and_Inventory_Optimization/
    ├── main.ipynb
    ├── main2.ipynb
    └── data/
        ├── raw/                 # Raw M5 files; not included in Git
        └── processed/
            ├── forecast_results.csv
            ├── aggregate_forecast.csv
            ├── inventory_recommendations.csv
            ├── category_inventory.csv
            └── store_inventory.csv
```

The dashboard reads the processed CSV files from both project folders.
Keep the folder names and relative paths unchanged unless you also
update `app.py`.

## Run the Dashboard Locally

### 1. Clone the repository

``` bash
git clone https://github.com/saudjabarali/Nexvia-Technologies-Data-Science-Internship.git
cd Nexvia-Technologies-Data-Science-Internship
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Launch the application

``` bash
streamlit run app.py
```

Streamlit will print a local address, typically `http://localhost:8501`.

> The dashboard requires the processed CSV files listed above. The raw
> M5 source files are not needed to view the dashboard when the
> processed outputs are present.

## Data Sources and Reproducibility

-   **DS-01:** Customer churn and customer-account dataset used for the
    customer analytics workflow.
-   **DS-02:** M5 forecasting dataset. Raw source files are not
    committed because of their size; processed dashboard outputs are
    stored under `data/processed/`.

The notebooks document the analytical workflows. The Streamlit
application presents selected processed results and visualizations; it
does not retrain the models when a visitor opens the dashboard.

## Limitations and Responsible Use

-   Results depend on the datasets, sampling choices, and assumptions
    described above.
-   Historical evaluation performance does not guarantee future
    performance.
-   CLV estimates, churn probabilities, uplift scores, demand forecasts,
    and inventory recommendations are analytical decision-support
    outputs.
-   Synthetic uplift assignment and simulated inventory inputs must not
    be mistaken for observed operational outcomes.
-   This project is intended for learning, demonstration, and analytical
    exploration---not production decision-making without further
    validation.

## Author

**Developed by Saud Jabar Ali**\
Data Science Intern --- Nexvia Technologies \| 2026

**Live Application:** [Nexvia Data Intelligence
Platform](https://nexvia-data-intelligence-saudjabarali.streamlit.app/)
