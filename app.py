import streamlit as st
import pandas as pd
import time
import kumoai.experimental.rfm as rfm
import os
import plotly.graph_objects as go
import numpy as np
from scipy import stats

st.set_page_config(page_title="Medicare Fraud Detection", layout="wide")
st.title("🚨🩺💊Medicare Provider Investigation Dashboard💊🩺🚨")

if 'validated' not in st.session_state:
    st.session_state.validated = False
    
# Initialize connection
@st.cache_resource
def init_connection():
    return st.connection("snowflake")

@st.cache_resource
def init_kumo():
    try:
        if not os.environ.get("KUMO_API_KEY"):
            rfm.authenticate()
        kumo_key = os.environ.get("KUMO_API_KEY")
        rfm.init(api_key=kumo_key)
        return 'success'
    except Exception as e:
        return 'fail'

@st.cache_data
def run_query_safe(query, max_retries=3):
    """Run query with automatic retry on connection failure"""
    for attempt in range(max_retries):
        try:
            conn = init_connection()
            df = conn.query(query, ttl=600)
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            if "Connection is closed" in str(e) and attempt < max_retries - 1:
                st.warning(f"Connection lost, retrying... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                st.cache_resource.clear()  # Clear the connection cache
            else:
                raise e


def create_timeseries_chart(historical_df, predicted_value, target_column, 
                           year_column='year', confidence_level=0.9, 
                            y_axis_title='Value'):
    """
    Create a time series chart with historical data and predicted value with statistical confidence interval.
    
    Parameters:
    - historical_df: DataFrame with historical data
    - predicted_value: Single predicted value (float/int)
    - target_column: Column name for the values to plot
    - year_column: Column name for the time/year data
    - confidence_level: Confidence level (default 0.95 = 95%)
    - title: Chart title
    - y_axis_title: Y-axis label
    """
    title = f"Time Series Prediction for {target_column}"
    
    # Create predicted dataframe
    predicted_df = pd.DataFrame({
        year_column: ['2024-01-01'],  # Adjust as needed
        target_column: [predicted_value],
        'data_type': ['Predicted']
    })
    
    # Prepare historical dataframe
    hist_df = historical_df[[year_column, target_column]].copy()
    hist_df['data_type'] = 'Historical'
    
    # Combine dataframes
    combined_df = pd.concat([hist_df, predicted_df], ignore_index=True)
    combined_df[year_column] = pd.to_datetime(combined_df[year_column])
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Add historical data
    historical_data = combined_df[combined_df['data_type'] == 'Historical']
    fig.add_trace(go.Scatter(
        x=historical_data[year_column],
        y=historical_data[target_column],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add predicted data with connection line
    predicted_data = combined_df[combined_df['data_type'] == 'Predicted']
    
    # Connect last historical point to predicted point with dotted line
    historical_sorted = historical_data.sort_values(year_column)
    last_historical = historical_sorted.iloc[-1]
    connection_df = pd.DataFrame({
        year_column: [last_historical[year_column], predicted_data.iloc[0][year_column]],
        target_column: [last_historical[target_column], predicted_data.iloc[0][target_column]]
    })
    
    fig.add_trace(go.Scatter(
        x=connection_df[year_column],
        y=connection_df[target_column],
        mode='lines',
        name='Prediction Connection',
        line=dict(color='red', width=2, dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=predicted_data[year_column],
        y=predicted_data[target_column],
        mode='markers',
        name='Predicted Value',
        marker=dict(color='red', size=10, symbol='diamond')
    ))
    
    # Calculate statistical confidence interval based on historical data
    historical_values = historical_df[target_column].values
    n = len(historical_values)
    
    # Calculate standard error of the mean
    std_dev = np.std(historical_values, ddof=1)  # Sample standard deviation
    std_error = std_dev / np.sqrt(n)
    
    # Calculate t-critical value for given confidence level
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Calculate margin of error
    margin_of_error = t_critical * std_error
    
    # For prediction interval (more appropriate for forecasting), we add uncertainty
    # Prediction interval accounts for both estimation uncertainty and natural variation
    prediction_std_error = std_dev * np.sqrt(1 + 1/n)
    prediction_margin = t_critical * prediction_std_error
    
    # Add statistical confidence interval
    predicted_val = predicted_data.iloc[0][target_column]
    
    # Use prediction interval for forecasting (more conservative and appropriate)
    upper_bound = predicted_val + prediction_margin
    lower_bound = max(0, predicted_val - prediction_margin)  # Ensure lower bound is not negative
    
    # Calculate actual margins for error bars (asymmetric if lower bound is constrained)
    upper_margin = upper_bound - predicted_val
    lower_margin = predicted_val - lower_bound
    
    # Add error bar for confidence interval (asymmetric if needed)
    fig.add_trace(go.Scatter(
        x=predicted_data[year_column],
        y=predicted_data[target_column],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[upper_margin],      # Upper error
            arrayminus=[lower_margin], # Lower error
            color='red',
            thickness=1.5,
            width=3
        ),
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0),
        name=f'{confidence_level*100:.0f}% Prediction Interval',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_axis_title,
        hovermode='x unified',
        yaxis=dict(
            range=[0, None]  # Anchor y-axis at 0, auto-scale the top
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
   
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.metric(value = f'{predicted_df[target_column][0]:,.2f}', label = '2024 Prediction')
    
    return fig



# Sidebar for additional options
with st.sidebar:
    st.header("Quick Actions")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
        
if 'kumo_connect' not in st.session_state:
    st.session_state.kumo_connect = init_kumo()
# Main application
tab1, tab2, tab3, tab4= st.tabs(["Home", "Kumo Predictions","Temporal Analysis","High Risk Providers"])


####################################################
## Tab 1
####################################################

with tab1:
    # Key Metrics Dashboard
        st.header("📊 System Overview")
        
        try:
            # Get summary statistics
            total_providers = run_query_safe("SELECT COUNT(DISTINCT NPI) as count FROM PROVIDERS")['count'][0]
            high_risk_count = run_query_safe("SELECT COUNT(*) as count FROM HIGH_RISK_PROVIDERS WHERE total_risk_score >= 5")['count'][0]
            excluded_providers = run_query_safe("SELECT COUNT(DISTINCT NPI) as count FROM EXCLUSIONS")['count'][0]
            
            avg_ba_risk = run_query_safe("SELECT AVG(risk_score) as avg_risk FROM PROVIDER_BILLING_ANOMALIES WHERE ID LIKE '%2023' ")['avg_risk'][0]
            avg_temp_risk = run_query_safe("SELECT AVG(temporal_risk_score) as avg_risk FROM TEMPORAL_PEER_ANALYSIS")['avg_risk'][0]
            avg_cross_risk = run_query_safe("SELECT AVG(cross_program_risk_score) as avg_risk FROM CROSS_PROGRAM_RISK WHERE ID LIKE '%2023' ")['avg_risk'][0]
            
            
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Providers Monitored",
                    value=f"{total_providers:,}",
                    delta="Active in system"
                )
                
                st.metric(
                    label="Average Temporal Risk",
                    value=f"{avg_temp_risk:.2}"
                )
            
            with col2:
                st.metric(
                    label="High Risk Providers",
                    value=f"{high_risk_count:,}",
                    delta=f"{(high_risk_count/total_providers)*100:.1f}% of total",
                    delta_color="inverse"
                )
            
                st.metric(
                    label="Average Cross Program Risk",
                    value=f"{avg_cross_risk:.2}"
                )
            with col3:
                st.metric(
                    label="Currently Excluded",
                    value=f"{excluded_providers:,}",
                    delta="OIG Exclusions",
                    delta_color="inverse"
                )
                st.metric(
                    label="Average Billing Anomalies Risk",
                    value=f"{avg_ba_risk:.2}"
                )
                
        except Exception as e:
            st.warning("Unable to load system metrics. Please check database connection.")
        
        # Features Overview
        st.header("🔍 Investigation Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 AI-Powered Predictions
            - **Kumo ML Models**: Advanced predictive analytics for risk assessment
            - **Cross-Program Analysis**: Identify suspicious patterns across Part B & Part D
            - **Temporal Forecasting**: Predict future billing anomalies and risk scores
            - **Real-time Scoring**: Dynamic risk assessment based on latest data
            """)
            
            st.markdown("""
            ### 📈 Billing Anomaly Detection
            - **Statistical Outlier Detection**: Z-score analysis vs peer groups
            - **Growth Pattern Analysis**: Identify suspicious payment increases
            - **Service Intensity Monitoring**: Track services per beneficiary
            - **Charge-to-Payment Ratios**: Monitor billing efficiency patterns
            """)
        
        with col2:
            st.markdown("""
            ### 💊 Cross-Program Risk Analysis
            - **Opioid Prescribing Patterns**: Monitor controlled substance prescriptions
            - **Specialty Mismatch Detection**: Flag inappropriate prescribing by specialty
            - **Geographic Risk Mapping**: Identify high-risk provider clusters
            - **Exclusion Network Analysis**: Track providers at excluded addresses
            """)
            
            st.markdown("""
            ### ⏰ Temporal & Peer Analysis
            - **Growth Trajectory Tracking**: Monitor provider evolution over time
            - **Peer Percentile Rankings**: Compare against similar providers
            - **New Provider Monitoring**: Special attention to recent market entrants
            - **Persistence Scoring**: Track sustained high-risk behavior
            """)
        
        # System Information
        st.header("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Data Sources
            - **Medicare Part B Claims** (2021-2023)
            - **Medicare Part D Prescriptions** (2021-2023)
            - **OIG Exclusion Lists** (Current)
            """)
            
            st.markdown("""
            ### 🎯 Risk Scoring Methodology
            - **Billing Risk**: Payment patterns, growth rates, peer comparison
            - **Cross-Program Risk**: Multi-program billing patterns, opioid prescribing
            - **Temporal Risk**: Historical trends, growth trajectories
            - **Combined Score**: Weighted algorithm considering all factors
            """)
            
        with col2:
            
            
            st.markdown("""
            ### 📞 Support & Contact
            - **Technical Support**: IT-Help@agency.gov
            - **Investigation Support**: fraud-unit@agency.gov
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Medicare Fraud Detection System v0.1 | Last Updated: August 2025</p>
            <p>⚠️ For Official Use Only - Contains Sensitive Investigation Data</p>
            <p>By: Bryce Drynan and Marcus Cooper </p>
            <p>Built for the KumoRFM Hackathon</p>
        </div>
        """, unsafe_allow_html=True)
####################################################
## Tab 2
####################################################           
        
with tab2:
    # ========================================
    # DATA LOADING AND MODEL INITIALIZATION
    # ========================================
    
    with st.spinner("Fetching Data For Model..."):
        # Load cross program risk data
        if 'cpr_table' not in st.session_state:
            st.session_state.cpr_table = run_query_safe("SELECT * FROM CROSS_PROGRAM_RISK")
        cross_prog_risk = st.session_state.cpr_table

        # Load temporal analysis data
        if 'ta_table' not in st.session_state:
            st.session_state.ta_table = run_query_safe("SELECT * FROM TEMPORAL_PEER_ANALYSIS")
        temporal_analysis = st.session_state.ta_table
                
        # Load billing anomalies data
        if 'ba_table' not in st.session_state:
            st.session_state.ba_table = run_query_safe("SELECT * FROM PROVIDER_BILLING_ANOMALIES")
        billing_anom = st.session_state.ba_table
    
        # Convert date columns to datetime
        cross_prog_risk['year'] = pd.to_datetime(cross_prog_risk['year'])
        billing_anom['year'] = pd.to_datetime(billing_anom['year'])   
        temporal_analysis['exclusion_date'] = pd.to_datetime(temporal_analysis['exclusion_date'])
    
    # Create local tables for Kumo RFM
    local_cpr = rfm.LocalTable(cross_prog_risk, name="cross_program_risk").infer_metadata()
    local_ta = rfm.LocalTable(temporal_analysis, name="temporal_analysis").infer_metadata()
    local_ba = rfm.LocalTable(billing_anom, name="billing_anomalies").infer_metadata()
    
    # Configure table metadata
    local_ta.primary_key = 'npi'
    local_ta['npi'].stype = 'ID'
    local_ta.time_column = None
    local_ta['temporal_risk_score'].stype = 'categorical'
    local_cpr['cross_program_risk_score'].stype = 'categorical'
    local_ba['risk_score'].stype = 'categorical'

    # Create graph and establish relationships
    graph = rfm.LocalGraph(tables=[local_ba, local_cpr, local_ta])
    graph.link(src_table=local_cpr, fkey='npi', dst_table=local_ta)
    graph.link(src_table=local_ba, fkey='npi', dst_table=local_ta)

    # Initialize Kumo RFM model
    with st.spinner("Model Loading..."):
        if 'model' not in st.session_state:
            st.session_state.model = rfm.KumoRFM(graph)
        model = st.session_state.model
        
    # ========================================
    # USER INTERFACE - FILTER OPTIONS
    # ========================================
    
    # Get available states for dropdown
    states = run_query_safe("SELECT DISTINCT STATE FROM CROSS_PROGRAM_RISK ORDER BY 1")['state']
    
    st.subheader("Select Filter Options")
    
    # State filter dropdown
    selected_state = st.selectbox(
        "Which State are you investigating?", 
        options=states,
        key="required_selectbox",
        help="Please select a state to filter by",
        index=None
    )
    
    # NPI filter text input with validation
    providers_avail = run_query_safe("SELECT NPI FROM PROVIDERS")
    specific_npi = st.text_input("Provide the NPI you are investigating: ")
    if specific_npi:
        try:
            if int(specific_npi) not in providers_avail['npi'].values:
                st.warning("Not a valid Provider, predictions will not work")
        except ValueError:
            st.warning("Please enter a valid numeric NPI")
    
    # Provider type filter dropdown
    p_types = run_query_safe("SELECT DISTINCT PROVIDER_TYPE FROM TEMPORAL_PEER_ANALYSIS")
    selected_p_type = st.selectbox(
        "Select the Provider Type you are investigating: ", 
        p_types, 
        index=None
    )
        
    # Number of results to display
    limit = st.number_input("Number of High Risk Providers to show", 1, 20, 5)
    
    # ========================================
    # ACTION BUTTONS
    # ========================================
    
    col1, col2, col3 = st.columns([.25, .25, .5])
    
    with col1:
        cpr_button = st.button("Predict Cross Program Risk")
        
    with col2:
        br_button = st.button("Predict Billing Risk")
    
    # ========================================
    # CROSS PROGRAM RISK PREDICTIONS
    # ========================================
    
    if cpr_button:
        # Build dynamic WHERE clause based on selected filters
        where_statement = "WHERE ID LIKE '%2023'"
        
        if selected_state:
            where_statement += f" AND STATE = '{selected_state}'"
        if specific_npi and specific_npi.strip():
            try:
                where_statement += f" AND NPI = {int(specific_npi)}"
            except ValueError:
                st.error("Invalid NPI format")
                st.stop()
        if selected_p_type:
            where_statement += f" AND PART_B_SPECIALTY = '{selected_p_type}'"
        
        # Get top risk providers based on filters
        curr_vals = run_query_safe(f"""
            SELECT ID, npi, cross_program_risk_score 
            FROM CROSS_PROGRAM_RISK 
            {where_statement} 
            ORDER BY cross_program_risk_score DESC 
            LIMIT {limit}
        """)

        # Process each high-risk provider
        for i in range(limit):
            with st.spinner(f"Predictions for {curr_vals['npi'][i]} loading..."):
                # Get historical data for current provider
                bad_docs = run_query_safe(f"""
                    SELECT * FROM CROSS_PROGRAM_RISK 
                    WHERE NPI = cast({curr_vals['npi'][i]} as int) 
                    ORDER BY 3 DESC
                """)
                
                # Define main prediction queries
                query = f"PREDICT SUM(cross_program_risk.combined_intensity_score,0,12,months) FOR temporal_analysis.npi = {curr_vals['npi'][i]}"
                query2 = f"PREDICT SUM(cross_program_risk.opioid_claim_rate,0,12,months) FOR temporal_analysis.npi = {curr_vals['npi'][i]}"
                query3 = f"PREDICT SUM(cross_program_risk.opioid_patient_rate,0,12,months) FOR temporal_analysis.npi = {curr_vals['npi'][i]}"
                
                # Define risk score prediction queries
                risk_score_queries = {
                    'sub_query1': f"PREDICT SUM(cross_program_risk.opioid_claim_rate_calc,0,12,months) > 0.3 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query2': f"PREDICT SUM(cross_program_risk.long_acting_opioid_rate_calc,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query3': f"PREDICT SUM(cross_program_risk.brand_preference_rate_calc,0,12,months) > 0.7 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query4': f"PREDICT SUM(cross_program_risk.antipsychotic_claims_count,0,12,months) > 500 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query5': f"PREDICT SUM(cross_program_risk.drug_to_medical_ratio_calc,0,12,months) > 5 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query6': f"PREDICT SUM(cross_program_risk.directly_excluded,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query7': f"PREDICT SUM(cross_program_risk.address_excluded_flag,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                }

                # Individual prediction queries with error handling
                df = None
                df2 = None
                df3 = None
                
                try:
                    df = model.predict(query, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with combined_intensity_score prediction: {str(e)}")
                
                try:
                    df2 = model.predict(query2, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with opioid_claim_rate prediction: {str(e)}")
                
                try:
                    df3 = model.predict(query3, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with opioid_patient_rate prediction: {str(e)}")
                   
                # Process risk score queries
                df4 = pd.DataFrame()
                for key in risk_score_queries:
                    try:
                        sub_df = model.predict(risk_score_queries[key], num_hops=6, run_mode='fast')
                        sub_df['Query'] = risk_score_queries[key]
                        df4 = pd.concat([df4, sub_df])
                    except Exception as e:
                        st.write(f"Error with risk score query {key}: {str(e)}")
                        continue
            
                # Display provider header and results
                st.header(f'High Cross Program Risk Provider: {bad_docs["npi"][0]}', divider=True)
                
                # Display risk score metric if available
                if not df4.empty:
                    st.metric(
                        value=df4['TARGET_PRED'].sum(), 
                        label='Projected 2024 Cross Program Risk Score', 
                        border=True
                    )
                else:
                    st.warning("Unable to calculate risk score due to prediction errors")
                
                # Display historical data
                st.dataframe(bad_docs)
                
                # Create time series charts only if predictions were successful
                if df is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'combined_intensity_score']], 
                            predicted_value=df['TARGET_PRED'][0], 
                            target_column='combined_intensity_score'
                        )
                    except Exception as e:
                        st.write(f"Error creating combined_intensity_score chart: {str(e)}")
                
                if df2 is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'opioid_claim_rate']], 
                            predicted_value=df2['TARGET_PRED'][0], 
                            target_column='opioid_claim_rate'
                        )
                    except Exception as e:
                        st.write(f"Error creating opioid_claim_rate chart: {str(e)}")
                
                if df3 is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'opioid_patient_rate']], 
                            predicted_value=df3['TARGET_PRED'][0], 
                            target_column='opioid_patient_rate'
                        )
                    except Exception as e:
                        st.write(f"Error creating opioid_patient_rate chart: {str(e)}")
                
                # Display raw predictions
                st.subheader("Raw Predictions for Score Creation")
                if not df4.empty:
                    st.dataframe(df4)
                else:
                    st.write("No risk score predictions available")
            
    # ========================================
    # BILLING RISK PREDICTIONS
    # ========================================
    
    if br_button:
        # Build dynamic WHERE clause based on selected filters
        where_statement = "WHERE ID LIKE '%2023'"
        
        if selected_state:
            where_statement += f" AND STATE = '{selected_state}'"
        if specific_npi and specific_npi.strip():
            try:
                where_statement += f" AND NPI = {int(specific_npi)}"
            except ValueError:
                st.error("Invalid NPI format")
                st.stop()
        if selected_p_type:
            where_statement += f" AND PROVIDER_TYPE = '{selected_p_type}'"
        
        # Get top billing risk providers based on filters
        curr_vals = run_query_safe(f"""
            SELECT ID, npi, risk_score 
            FROM PROVIDER_BILLING_ANOMALIES 
            {where_statement} 
            ORDER BY risk_score DESC 
            LIMIT {limit}
        """)
        
        # Process each high billing risk provider
        for i in range(limit):
            with st.spinner(f"Predictions for {curr_vals['npi'][i]} loading..."):
                # Get historical data for current provider
                bad_docs = run_query_safe(f"""
                    SELECT * FROM PROVIDER_BILLING_ANOMALIES 
                    WHERE NPI = cast({curr_vals['npi'][i]} as int) 
                    ORDER BY 3 DESC
                """)
                
                # Define main prediction queries
                query = f'PREDICT SUM(billing_anomalies.total_medicare_reimbursement,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                query2 = f'PREDICT SUM(billing_anomalies.payment_per_service,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                query3 = f'PREDICT SUM(billing_anomalies.services_per_beneficiary,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                query4 = f'PREDICT SUM(billing_anomalies.part_d_claims,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                
                # Define risk score prediction queries
                risk_score_queries = {
                    'sub_query1': f"PREDICT SUM(billing_anomalies.payment_zscore,0,12,months) > 2 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query2': f"PREDICT SUM(billing_anomalies.payment_growth_rate,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query3': f"PREDICT SUM(billing_anomalies.service_growth_rate,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query4': f"PREDICT SUM(billing_anomalies.opioid_prescribing_rate,0,12,months) > 0.3 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query5': f"PREDICT SUM(billing_anomalies.opioid_prescriber_rate,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query6': f"PREDICT SUM(billing_anomalies.charge_to_payment_ratio,0,12,months) > 3 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query7': f"PREDICT SUM(billing_anomalies.oig_excluded_flag,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals['npi'][i]}",
                    'sub_query8': f"PREDICT SUM(billing_anomalies.payment_outlier_flag,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals['npi'][i]}", 
                    'sub_query9': f"PREDICT SUM(billing_anomalies.payment_zscore,0,12,months) < 2 FOR temporal_analysis.npi = {curr_vals['npi'][i]}"      
                }

                # Individual prediction queries with error handling
                df = None
                df2 = None
                df3 = None
                df4 = None
                
                try:
                    df = model.predict(query, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with total_medicare_reimbursement prediction: {str(e)}")
                
                try:
                    df2 = model.predict(query2, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with payment_per_service prediction: {str(e)}")
                
                try:
                    df3 = model.predict(query3, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with services_per_beneficiary prediction: {str(e)}")
                
                try:
                    df4 = model.predict(query4, num_hops=6, run_mode='fast')
                except Exception as e:
                    st.write(f"Error with part_d_claims prediction: {str(e)}")
                
                # Process risk score queries
                df5 = pd.DataFrame()
                for key in risk_score_queries:
                    try:
                        sub_df = model.predict(risk_score_queries[key], num_hops=6, run_mode='fast')
                        sub_df['Query'] = risk_score_queries[key]
                        df5 = pd.concat([df5, sub_df])
                    except Exception as e:
                        st.write(f"Error with risk score query {key}: {str(e)}")
                        continue
            
                # Display provider header and results
                st.header(f'High Billing Risk Provider: {bad_docs["npi"][0]}', divider=True)
                
                # Display risk score metric if available
                if not df5.empty:
                    st.metric(
                        value=df5['TARGET_PRED'].sum(), 
                        label='Projected 2024 Billing Risk Score', 
                        border=True
                    )
                else:
                    st.warning("Unable to calculate risk score due to prediction errors")
                
                # Display historical data
                st.dataframe(bad_docs)
                
                # Create time series charts only if predictions were successful
                if df is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'total_medicare_reimbursement']], 
                            predicted_value=df['TARGET_PRED'][0], 
                            target_column='total_medicare_reimbursement'
                        )
                    except Exception as e:
                        st.write(f"Error creating total_medicare_reimbursement chart: {str(e)}")
                
                if df2 is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'payment_per_service']], 
                            predicted_value=df2['TARGET_PRED'][0], 
                            target_column='payment_per_service'
                        )
                    except Exception as e:
                        st.write(f"Error creating payment_per_service chart: {str(e)}")
                
                if df3 is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'services_per_beneficiary']], 
                            predicted_value=df3['TARGET_PRED'][0], 
                            target_column='services_per_beneficiary'
                        )
                    except Exception as e:
                        st.write(f"Error creating services_per_beneficiary chart: {str(e)}")
                
                if df4 is not None:
                    try:
                        create_timeseries_chart(
                            historical_df=bad_docs[['year', 'part_d_claims']], 
                            predicted_value=df4['TARGET_PRED'][0], 
                            target_column='part_d_claims'
                        )
                    except Exception as e:
                        st.write(f"Error creating part_d_claims chart: {str(e)}")
                
                # Display raw predictions
                st.subheader("Raw Predictions for Score Creation")
                if not df5.empty:
                    st.dataframe(df5)
                else:
                    st.write("No risk score predictions available")
               
            
####################################################
## Tab 3
####################################################
with tab3:
    # ========================================
    # TEMPORAL ANALYSIS INTERFACE
    # ========================================
    
    st.header("Temporal Analysis")
    
    # ========================================
    # FILTER OPTIONS
    # ========================================
    
    # Growth trajectory filter dropdown
    growth_type = st.selectbox(
        "Growth Trajectory",
        ["All", "Suspicious", "Growing", "Stable", "New_Provider"],
        help="Select the growth pattern to analyze"
    )
    
    # Get available states for dropdown
    states = run_query_safe("SELECT DISTINCT STATE FROM CROSS_PROGRAM_RISK ORDER BY 1")['state']
    
    # State filter - direct selectbox (no checkbox)
    selected_state = st.selectbox(
        "Which State are you investigating?", 
        options=states,
        key="temporal_state_selectbox",
        help="Please select a state to filter by",
        index=None
    )
    
    # NPI filter - direct text input (no checkbox) with validation
    providers_avail = run_query_safe("SELECT NPI FROM PROVIDERS")
    specific_npi = st.text_input(
        "Provide the NPI you are investigating: ",
        key="temporal_npi_input",
        help="Enter a specific NPI to analyze"
    )
    
    # Validate NPI input if provided
    if specific_npi:
        try:
            if int(specific_npi) not in providers_avail['npi'].values:
                st.warning("Not a valid Provider, analysis may not work properly")
        except ValueError:
            st.warning("Please enter a valid numeric NPI")
    
    # ========================================
    # TEMPORAL ANALYSIS EXECUTION
    # ========================================
    
    if st.button("Load Temporal Analysis", key="temporal"):
        with st.spinner("Loading temporal analysis..."):
            # Build dynamic WHERE clause based on selected filters
            where_clause = "WHERE 1=1"
            
            # Apply growth trajectory filter
            if growth_type != 'All':
                where_clause += f" AND growth_trajectory = '{growth_type}'"
            
            # Apply state filter if selected
            if selected_state:
                where_clause += f" AND STATE = '{selected_state}'"
            
            # Apply NPI filter if provided and valid
            if specific_npi and specific_npi.strip():
                try:
                    where_clause += f" AND NPI = {int(specific_npi)}"
                except ValueError:
                    st.error("Invalid NPI format - skipping NPI filter")
            
            # Construct temporal analysis query
            query = f"""
            SELECT 
                NPI,
                provider_name_address,
                growth_trajectory,
                payment_2021,
                payment_2022,
                payment_2023,
                growth_21_to_22,
                growth_22_to_23,
                peer_percentile_range,
                temporal_risk_score
            FROM MEDICARE_DATA.MODEL_READY.TEMPORAL_PEER_ANALYSIS
            {where_clause}
            ORDER BY temporal_risk_score DESC
            LIMIT 50
            """
            
            # Execute query and display results
            try:
                df = run_query_safe(query)
                
                # Display results if data is available
                if not df.empty:
                    st.success(f"Found {len(df)} providers matching your criteria")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Average Risk Score", 
                            f"{df['temporal_risk_score'].mean():.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Highest Risk Score", 
                            f"{df['temporal_risk_score'].max():.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Total Providers", 
                            len(df)
                        )
                    
                    with col4:
                        if 'payment_2023' in df.columns:
                            st.metric(
                                "Total 2023 Payments", 
                                f"${df['payment_2023'].sum():,.0f}"
                            )
                
                else:
                    st.warning("No providers found matching your criteria. Please adjust your filters and try again.")
                    
            except Exception as e:
                st.error(f"Error executing temporal analysis: {str(e)}")
                st.info("Please check your filter selections and try again.")
            
####################################################
## Tab 4
####################################################  
with tab4:
    # ========================================
    # HIGH RISK PROVIDERS OVERVIEW INTERFACE
    # ========================================
    
    st.header("High Risk Providers Overview")
    
    # ========================================
    # FILTER CONFIGURATION
    # ========================================
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Risk score threshold slider
        risk_threshold = st.slider(
            "Minimum Risk Score", 
            min_value=0, 
            max_value=10, 
            value=3,
            help="Filter providers by minimum total risk score"
        )
    
    with col2:
        # Maximum number of providers to display
        limit = st.number_input(
            "Max number of providers to show", 
            min_value=10, 
            max_value=100, 
            value=20,
            help="Limit the number of results displayed"
        )
    
    with col3:
        # Include excluded providers option - simplified to single checkbox
        include_excluded = st.checkbox(
            "Include Excluded Providers",
            value=False,
            help="Check to include providers who are currently excluded"
        )
    
    # ========================================
    # HIGH RISK PROVIDERS ANALYSIS
    # ========================================
    
    if st.button("Load High Risk Providers", key="high_risk"):
        with st.spinner("Loading high risk providers data..."):
            
            # Build query with proper boolean logic for excluded providers
            excluded_filter = "1" if include_excluded else "0"
            
            query = f"""
            SELECT 
                NPI,
                provider_info,
                STATE,
                PROVIDER_TYPE,
                billing_risk_score,
                cross_program_risk_score,
                temporal_risk_score,
                total_risk_score,
                currently_excluded,
                opioid_rate / 100 as opioid_rate
            FROM MEDICARE_DATA.MODEL_READY.HIGH_RISK_PROVIDERS
            WHERE total_risk_score >= {risk_threshold} 
                AND currently_excluded <= {excluded_filter}
            ORDER BY total_risk_score DESC
            LIMIT {limit}
            """
            
            try:
                # Execute query and retrieve data
                df = run_query_safe(query)
                
                # Check if data was retrieved successfully
                if not df.empty:
                    st.success(f"Found {len(df)} high risk providers matching your criteria")
                    
                    # ========================================
                    # SUMMARY METRICS DISPLAY
                    # ========================================
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Providers", 
                            len(df),
                            help="Number of providers in results"
                        )
                    
                    with col2:
                        excluded_count = df['currently_excluded'].sum() if 'currently_excluded' in df.columns else 0
                        st.metric(
                            "Currently Excluded", 
                            int(excluded_count),
                            help="Number of providers currently excluded from Medicare"
                        )
                    
                    with col3:
                        avg_risk = df['total_risk_score'].mean()
                        st.metric(
                            "Avg Risk Score", 
                            f"{avg_risk:.2f}",
                            help="Average total risk score of displayed providers"
                        )
                    
                    with col4:
                        # Calculate high opioid prescribers (>30% opioid prescribing rate)
                        high_opioid = (df['opioid_rate'] > 0.3).sum() if 'opioid_rate' in df.columns else 0
                        st.metric(
                            "High Opioid Prescribers", 
                            int(high_opioid),
                            help="Providers with >30% opioid prescribing rate"
                        )
                    
                    # ========================================
                    # DETAILED DATA DISPLAY
                    # ========================================
                    
                    st.subheader("Provider Details")
                    
                    # Display data with enhanced formatting
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "opioid_rate": st.column_config.ProgressColumn(
                                "Opioid Rate",
                                min_value=0,
                                max_value=1,
                                help="Percentage of opioid prescriptions"
                            ),
                            "total_risk_score": st.column_config.NumberColumn(
                                "Total Risk Score",
                                format="%.2f",
                                help="Combined risk score across all categories"
                            ),
                            "billing_risk_score": st.column_config.NumberColumn(
                                "Billing Risk",
                                format="%.2f",
                                help="Risk score for billing anomalies"
                            ),
                            "cross_program_risk_score": st.column_config.NumberColumn(
                                "Cross Program Risk",
                                format="%.2f",
                                help="Risk score for cross-program patterns"
                            ),
                            "temporal_risk_score": st.column_config.NumberColumn(
                                "Temporal Risk",
                                format="%.2f",
                                help="Risk score for temporal patterns"
                            ),
                            "currently_excluded": st.column_config.CheckboxColumn(
                                "Excluded",
                                help="Currently excluded from Medicare"
                            ),
                            "NPI": st.column_config.TextColumn(
                                "NPI",
                                help="National Provider Identifier"
                            )
                        }
                    )
                    
                    # ========================================
                    # ADDITIONAL INSIGHTS
                    # ========================================
                    
                    st.subheader("Risk Distribution Analysis")
                    
                    # Create risk distribution summary
                    risk_ranges = {
                        "Low Risk (0-3)": len(df[df['total_risk_score'] <= 3]),
                        "Medium Risk (3-6)": len(df[(df['total_risk_score'] > 3) & (df['total_risk_score'] <= 6)]),
                        "High Risk (6-8)": len(df[(df['total_risk_score'] > 6) & (df['total_risk_score'] <= 8)]),
                        "Very High Risk (8+)": len(df[df['total_risk_score'] > 8])
                    }
                    
                    # Display risk distribution
                    col1, col2, col3, col4 = st.columns(4)
                    for i, (risk_level, count) in enumerate(risk_ranges.items()):
                        with [col1, col2, col3, col4][i]:
                            st.metric(risk_level, count)
                    
                    # State-wise breakdown if multiple states present
                    if 'STATE' in df.columns and df['STATE'].nunique() > 1:
                        st.subheader("State-wise Breakdown")
                        state_summary = df.groupby('STATE').agg({
                            'NPI': 'count',
                            'total_risk_score': 'mean',
                            'currently_excluded': 'sum'
                        }).round(2)
                        state_summary.columns = ['Provider Count', 'Avg Risk Score', 'Excluded Count']
                        st.dataframe(state_summary, use_container_width=True)
                    
                    # ========================================
                    # PROVIDER TYPE RISK ANALYSIS
                    # ========================================
                    
                    if 'PROVIDER_TYPE' in df.columns and df['PROVIDER_TYPE'].nunique() > 1:
                        st.subheader("🏥 Provider Type Risk Analysis")
                        
                        # Calculate provider type statistics
                        provider_type_analysis = df.groupby('PROVIDER_TYPE').agg({
                            'NPI': 'count',
                            'total_risk_score': ['mean', 'max'],
                            'billing_risk_score': 'mean',
                            'cross_program_risk_score': 'mean',
                            'temporal_risk_score': 'mean',
                            'currently_excluded': 'sum',
                            'opioid_rate': 'mean'
                        }).round(2)
                        
                        # Flatten column names for better readability
                        provider_type_analysis.columns = [
                            'Providers', 'Avg Risk', 'Max Risk',
                            'Billing Risk', 'Cross Program', 'Temporal Risk',
                            'Excluded', 'Opioid Rate'
                        ]
                        
                        # Sort by average total risk score descending
                        provider_type_analysis = provider_type_analysis.sort_values('Avg Risk', ascending=False)
                        
                        st.dataframe(
                            provider_type_analysis, 
                            use_container_width=True,
                            column_config={
                                "Opioid Rate": st.column_config.ProgressColumn(
                                    "Opioid Rate",
                                    min_value=0.0,
                                    max_value=1.0,
                                    help="Average opioid prescribing rate for this provider type"
                                ),
                                "Avg Risk": st.column_config.NumberColumn(
                                    "Avg Risk",
                                    format="%.1f",
                                    help="Average total risk score"
                                ),
                                "Max Risk": st.column_config.NumberColumn(
                                    "Max Risk", 
                                    format="%.1f",
                                    help="Highest individual risk score in this category"
                                ),
                                "Billing Risk": st.column_config.NumberColumn(
                                    "Billing Risk",
                                    format="%.1f"
                                ),
                                "Cross Program": st.column_config.NumberColumn(
                                    "Cross Program", 
                                    format="%.1f"
                                ),
                                "Temporal Risk": st.column_config.NumberColumn(
                                    "Temporal Risk",
                                    format="%.1f"
                                )
                            }
                        )
                        
                        # Show top 3 highest risk provider types with clear formatting
                        st.write("**🎯 Highest Risk Provider Types:**")
                        for i in range(min(3, len(provider_type_analysis))):
                            provider_type = provider_type_analysis.index[i]
                            avg_risk = provider_type_analysis.iloc[i]['Avg Risk']
                            provider_count = provider_type_analysis.iloc[i]['Providers']
                            
                            if avg_risk >= 6:
                                risk_level = "🔴 VERY HIGH"
                            elif avg_risk >= 4:
                                risk_level = "🟡 HIGH"
                            else:
                                risk_level = "🟢 MODERATE"
                            
                            st.write(f"**{i+1}.** {provider_type} - {risk_level} Risk ({avg_risk:.1f}) - {provider_count} providers")
                    
                    # ========================================
                    # RISK INSIGHTS DASHBOARD
                    # ========================================
                    
                    st.subheader("📊 Risk Pattern Insights")
                    
                    # Create three columns for different insights
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**🎯 Risk Components**")
                        
                        # Calculate average contribution of each risk component
                        if all(col in df.columns for col in ['billing_risk_score', 'cross_program_risk_score', 'temporal_risk_score']):
                            avg_billing = df['billing_risk_score'].mean()
                            avg_cross_program = df['cross_program_risk_score'].mean()
                            avg_temporal = df['temporal_risk_score'].mean()
                            
                            risk_components = {
                                "Billing": avg_billing,
                                "Cross Program": avg_cross_program,
                                "Temporal": avg_temporal
                            }
                            
                            # Find dominant risk factor
                            dominant_risk = max(risk_components, key=risk_components.get)
                            
                            st.metric("🏆 Primary Risk Driver", dominant_risk, f"{risk_components[dominant_risk]:.1f}")
                            
                            # Show all components in a clean format
                            for risk_type, score in sorted(risk_components.items(), key=lambda x: x[1], reverse=True):
                                if score >= 4:
                                    emoji = "🔴"
                                elif score >= 2:
                                    emoji = "🟡"
                                else:
                                    emoji = "🟢"
                                st.write(f"{emoji} {risk_type}: **{score:.1f}**")
                    
                    with col2:
                        st.write("**⚠️ Alert Indicators**")
                        
                        # Calculate high-risk thresholds and counts
                        very_high_risk = len(df[df['total_risk_score'] >= 7])
                        high_opioid_prescribers = len(df[df['opioid_rate'] > 0.4]) if 'opioid_rate' in df.columns else 0
                        multi_risk_providers = 0
                        
                        # Count providers with high scores across multiple categories
                        if all(col in df.columns for col in ['billing_risk_score', 'cross_program_risk_score', 'temporal_risk_score']):
                            high_billing = df['billing_risk_score'] >= 3
                            high_cross = df['cross_program_risk_score'] >= 3
                            high_temporal = df['temporal_risk_score'] >= 3
                            multi_risk_providers = len(df[high_billing & high_cross & high_temporal])
                        
                        st.metric("🚨 Critical Risk (7+)", very_high_risk)
                        st.metric("💊 High Opioid (40%+)", high_opioid_prescribers)
                        st.metric("🎯 Multi-Category Risk", multi_risk_providers)
                        
                        # Show percentage of total
                        total_providers = len(df)
                        if very_high_risk > 0:
                            critical_pct = (very_high_risk / total_providers) * 100
                            st.write(f"*{critical_pct:.0f}% of providers are critical risk*")
                    
                    with col3:
                        st.write("**📈 Risk Distribution**")
                        
                        # Calculate risk distribution
                        low_risk = len(df[df['total_risk_score'] < 3])
                        medium_risk = len(df[(df['total_risk_score'] >= 3) & (df['total_risk_score'] < 6)])
                        high_risk = len(df[(df['total_risk_score'] >= 6) & (df['total_risk_score'] < 8)])
                        very_high_risk = len(df[df['total_risk_score'] >= 8])
                        
                        st.metric("🟢 Low (0-3)", low_risk)
                        st.metric("🟡 Medium (3-6)", medium_risk) 
                        st.metric("🟠 High (6-8)", high_risk)
                        st.metric("🔴 Very High (8+)", very_high_risk)
                        
                        # Show highest concentration
                        risk_counts = {"Low": low_risk, "Medium": medium_risk, "High": high_risk, "Very High": very_high_risk}
                        dominant_category = max(risk_counts, key=risk_counts.get)
                        st.write(f"*Most providers: **{dominant_category} Risk***")
                    
                    # Special alert for multi-category high risk providers
                    if multi_risk_providers > 0:
                        st.error(f"🚨 **PRIORITY ALERT**: {multi_risk_providers} providers show high risk across ALL categories and require immediate investigation!")
                    
                    # ========================================
                    # ACTIONABLE RECOMMENDATIONS
                    # ========================================
                    
                    st.subheader("💡 Recommended Actions")
                    
                    # Create clearer, more actionable recommendations
                    recommendations = []
                    priority_actions = []
                    
                    # Generate priority actions (red alerts)
                    if multi_risk_providers > 0:
                        priority_actions.append(f"**URGENT**: Investigate {multi_risk_providers} providers with high risk across all categories")
                    
                    if very_high_risk > 5:
                        priority_actions.append(f"**URGENT**: {very_high_risk} providers have critical risk scores (7+) - immediate review needed")
                    
                    # Generate standard recommendations (yellow/blue)
                    if excluded_count > 0:
                        recommendations.append(f"**Monitor**: {int(excluded_count)} excluded providers still appear high-risk - verify exclusion status")
                    
                    if high_opioid > len(df) * 0.25:  # If >25% have high opioid rates
                        recommendations.append(f"**Review**: {high_opioid} providers show concerning opioid prescribing patterns (>40%)")
                    
                    if 'PROVIDER_TYPE' in df.columns and df['PROVIDER_TYPE'].nunique() > 1:
                        top_risk_type = provider_type_analysis.index[0]
                        top_risk_score = provider_type_analysis.iloc[0]['Avg Risk']
                        recommendations.append(f"**Focus**: Target '{top_risk_type}' providers (highest avg risk: {top_risk_score:.1f})")
                    
                    # Display priority actions first
                    if priority_actions:
                        st.write("**🚨 Priority Actions:**")
                        for action in priority_actions:
                            st.error(f"• {action}")
                        st.write("")  # Add spacing
                    
                    # Display standard recommendations
                    if recommendations:
                        st.write("**📋 Additional Recommendations:**")
                        for rec in recommendations:
                            st.info(f"• {rec}")
                    
                    # Show success message if no major issues
                    if not priority_actions and not recommendations:
                        st.success("✅ **Good News**: No high-priority risk patterns detected. Current risk levels appear manageable.")
                        st.write("Continue regular monitoring and consider lowering risk thresholds to identify emerging patterns.")
                
                else:
                    st.warning("No providers found matching your criteria. Please adjust your filters and try again.")
                    st.info("Try lowering the minimum risk score or increasing the provider limit.")
                    
            except Exception as e:
                st.error(f"Error loading high risk providers data: {str(e)}")
                st.info("Please check your database connection and try again.")
                
                
                
