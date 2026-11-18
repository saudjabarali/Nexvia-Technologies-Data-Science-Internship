# DS-02: Nationwide Demand Forecasting and Inventory Optimization

## Project Overview

This project focuses on developing an intelligent demand forecasting and inventory optimization system capable of predicting product demand across multiple stores and geographical locations.

The system aims to help businesses make better inventory management decisions by forecasting future product demand, minimizing stockouts, reducing excess inventory, and optimizing replenishment quantities.

The project will use historical retail sales data to investigate demand patterns and implement machine learning and deep learning forecasting techniques.

## Objectives

- Analyze historical retail sales data across multiple stores and products.
- Identify seasonal trends, demand patterns, and sales fluctuations.
- Develop accurate demand forecasting models.
- Implement hierarchical forecasting across different geographical and product levels.
- Generate prediction intervals to quantify forecasting uncertainty.
- Optimize inventory quantities using mathematical optimization.
- Develop an interactive dashboard for visualizing forecasts and inventory recommendations.

## Dataset

**Primary Dataset:** Walmart M5 Forecasting Dataset

The M5 dataset contains historical Walmart sales data and supporting information, including:

- Daily product sales.
- Store and department information.
- Product categories.
- Calendar and holiday information.
- Selling prices.

Dataset source: Kaggle – M5 Forecasting Accuracy Competition.

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Forecasting | Prophet |
| Machine Learning | LightGBM |
| Deep Learning | PyTorch Forecasting (TFT) |
| Time Series Framework | Darts |
| Optimization | Google OR-Tools |
| Dashboard | Plotly Dash |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## Project Workflow

### 1. Dataset Acquisition and Understanding
- Download the Walmart M5 dataset.
- Understand the dataset structure.
- Examine product, store, department, and geographical hierarchies.
- Perform initial data quality checks.

### 2. Exploratory Data Analysis (EDA)
- Analyze historical sales patterns.
- Identify seasonal trends.
- Examine demand variations across stores and products.
- Investigate the effects of holidays and pricing.

### 3. Feature Engineering
- Lag features.
- Rolling averages.
- Calendar features.
- Holiday indicators.
- Price-related features.
- Seasonal features.

### 4. Demand Forecasting

Implement and compare multiple forecasting approaches:

- Prophet
- LightGBM
- Temporal Fusion Transformer (TFT)

Establish baseline forecasting models before evaluating advanced approaches.

### 5. Hierarchical Forecasting
- Generate forecasts at different aggregation levels.
- Investigate hierarchical forecast reconciliation.
- Maintain consistency between product, department, store, and geographical forecasts.

### 6. Prediction Intervals
- Estimate forecasting uncertainty.
- Generate prediction intervals.
- Evaluate forecast reliability.

### 7. Inventory Optimization
- Convert demand forecasts into inventory recommendations.
- Determine optimal replenishment quantities.
- Minimize inventory costs and stockout risks.
- Implement optimization using Google OR-Tools.

### 8. Interactive Dashboard

Develop an interactive dashboard using Plotly Dash to display:

- Historical sales trends.
- Predicted demand.
- Forecast comparisons.
- Inventory recommendations.
- Stockout risks.
- Inventory optimization insights.

## Repository Structure

```text
DS-02_Demand_Forecasting_and_Inventory_Optimization/
│
├── README.md
│
├── data/
│
├── notebooks/
│   ├── 01_data_loading_and_hierarchy.ipynb
│   ├── 02_eda_and_feature_engineering.ipynb
│   ├── 03_baseline_and_prophet.ipynb
│   ├── 04_lightgbm_forecasting.ipynb
│   ├── 05_deep_learning_tft.ipynb
│   ├── 06_hierarchical_reconciliation_and_intervals.ipynb
│   └── 07_inventory_optimization_and_dashboard.ipynb
│
├── src/
│
├── app/
│
└── reports/
```

## Evaluation Metrics

Forecasting models will be evaluated using appropriate metrics, including:

- RMSE – Root Mean Squared Error.
- WAPE – Weighted Absolute Percentage Error.
- RMSSE – Root Mean Squared Scaled Error.

Model performance will be compared using a consistent time-based validation strategy.

## Project Status

**Status:** Initial Setup and Environment Configuration

Implementation has not yet started.

## Future Updates

This README will be expanded throughout development to include:

- Dataset preprocessing methodology.
- Feature engineering techniques.
- Model architectures and configurations.
- Experimental results and model comparisons.
- Inventory optimization methodology.
- Dashboard screenshots.
- Installation and execution instructions.
- Final conclusions and limitations.