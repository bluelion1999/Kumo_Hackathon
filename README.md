# Medicare Fraud Detection Dashboard

A Streamlit-based application for detecting Medicare fraud using KumoAI's RFM (Relational Foundation Model) for predictive analytics on Medicare Part B and Part D data.

**Built by:** Bryce Drynan and Marcus Cooper  
**Built for:** KumoRFM Hackathon

## Overview

This application provides an investigation dashboard for Medicare fraud detection, featuring:
- AI-powered predictions using Kumo ML models
- Cross-program risk analysis between Part B and Part D
- Temporal analysis of provider billing patterns
- High-risk provider identification

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── gold_tables.sql          # SQL for creating analysis tables
├── silver_tables.sql        # SQL for intermediate data transformations
├── model_ready_table_def.sql # Table definitions for model-ready data
├── raw_table_def.sql        # Raw data table definitions
├── .gitignore               # Git ignore configuration
└── LICENSE                  # MIT License
```

## Features

### 1. System Overview (Home Tab)
- Key metrics dashboard showing total providers, high-risk counts, and exclusions
- Average risk scores across different categories (temporal, cross-program, billing anomalies)
- Investigation capabilities overview
- System information and methodology

### 2. Kumo Predictions Tab
- Predictive analytics for Cross Program Risk and Billing Risk
- Time series forecasting for 2024 based on 2021-2023 historical data
- Confidence intervals for predictions
- Filtering options by state, NPI, and provider type

### 3. Temporal Analysis Tab
- Growth trajectory classification (Suspicious, Growing, Stable, New Provider)
- Year-over-year payment and service analysis
- Peer percentile rankings
- Risk scoring based on temporal patterns

### 4. High Risk Providers Tab
- Configurable risk score thresholds
- Provider filtering and exclusion status
- Comprehensive risk metrics display
- Opioid prescribing rate visualization

## Data Sources

The application analyzes:
- **Medicare Part B Claims** (2021-2023)
- **Medicare Part D Prescriptions** (2021-2023)
- **OIG Exclusion Lists** (Current)

## Risk Scoring Components

- **Billing Risk**: Payment patterns, growth rates, peer comparison
- **Cross-Program Risk**: Multi-program billing patterns, opioid prescribing
- **Temporal Risk**: Historical trends, growth trajectories
- **Combined Score**: Weighted algorithm considering all factors

## Database Schema

The application uses a Snowflake database with the following key tables:
- `MEDICARE_DATA.MODEL_READY.PROVIDER_BILLING_ANOMALIES`
- `MEDICARE_DATA.MODEL_READY.CROSS_PROGRAM_RISK`
- `MEDICARE_DATA.MODEL_READY.TEMPORAL_PEER_ANALYSIS`
- `MEDICARE_DATA.MODEL_READY.HIGH_RISK_PROVIDERS`
- `MEDICARE_DATA.MODEL_READY.EXCLUSIONS`

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
- Set up Kumo API key (will prompt for authentication if not set)
- Configure Snowflake connection in Streamlit secrets

3. Run the application:
```bash
streamlit run app.py
```

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **Snowflake**: Data warehouse
- **KumoAI**: Machine learning predictions
- **Plotly**: Data visualization
- **NumPy/SciPy**: Statistical analysis

## License

MIT License - See LICENSE file for details

## Version

Medicare Fraud Detection System v0.1
