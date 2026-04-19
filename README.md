# TrendLens

TrendLens is a data-driven insight generation system designed to transform structured datasets into interpretable analysis through statistical profiling, visual exploration, clustering, and automated insight reporting.

The system is built to bridge the gap between raw data and structured understanding, enabling users to explore patterns, relationships, and trends through a clean and unified interface.

---

## Overview

TrendLens provides an end-to-end analytical workflow:
- Dataset ingestion and preprocessing
- Data quality assessment
- Statistical analysis
- Visual exploration
- Segmentation through clustering
- Automated generation of written insights

Rather than focusing only on visualisation, the system emphasises **structured analysis and interpretability**, aligning with real-world use cases where data must support decision-making.

---

## Features

### Dataset Profiling
- Automatic detection of dataset shape, column types, and structure
- Missing value analysis and data quality indicators
- Clean tabular preview of data

### Statistical Analysis
- Summary statistics for all numeric features
- Distribution analysis using histograms and box plots
- Identification of variability and dominant features

### Visual Exploration
- Scatter plots for feature relationships
- Correlation heatmaps for pattern discovery
- Grouped comparisons across categorical variables
- Time-based trend analysis (when datetime fields are present)

### Clustering and Segmentation
- K-Means clustering on selected numeric features
- Standardisation of inputs for meaningful grouping
- Visual cluster projection for exploratory segmentation

### Insight Generation
- Automated plain-language summaries of key findings
- Identification of strong correlations and dominant variables
- Detection of potential data quality issues

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn

---

## Running the App

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
