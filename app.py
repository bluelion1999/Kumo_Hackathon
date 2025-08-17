import streamlit as st
import pandas as pd
import time
import kumoai.experimental.rfm as rfm
import os
import plotly.graph_objects as go
import numpy as np
from scipy import stats

st.set_page_config(page_title="Medicare Fraud Detection", layout="wide")
st.write("By: Bryce Drynan and Marcus Cooper")
st.write("Built for the KumoRFM Hackathon")
st.title("🚨🩺💊Medicare Provider Investigation Dashboard💊🩺🚨")

if 'validated' not in st.session_state:
    st.session_state.validated = False
    
# Initialize connection
@st.cache_resource
def init_connection():
    return st.connection("snowflake")

kumo_key = os.getenv('KUMO_ACCESS_KEY_ID')
rfm.init(api_key=kumo_key)

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


# Main application
tab1, tab2, tab3, tab4= st.tabs(["Home", "Kumo Predictions","Temporal Analysis","High Risk Providers"])
with tab1:
    # Key Metrics Dashboard
    st.header("📊 System Overview")
    
    try:
        # Get summary statistics
        total_providers = run_query_safe("SELECT COUNT(DISTINCT NPI) as count FROM PROVIDERS")['count'][0]
        high_risk_count = run_query_safe("SELECT COUNT(*) as count FROM HIGH_RISK_PROVIDERS WHERE total_risk_score >= 5")['count'][0]
        excluded_providers = run_query_safe("SELECT COUNT(DISTINCT NPI) as count FROM EXCLUSIONS")['count'][0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Providers Monitored",
                value=f"{total_providers:,}",
                delta="Active in system"
            )
        
        with col2:
            st.metric(
                label="High Risk Providers",
                value=f"{high_risk_count:,}",
                delta=f"{(high_risk_count/total_providers)*100:.1f}% of total",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="Currently Excluded",
                value=f"{excluded_providers:,}",
                delta="OIG Exclusions",
                delta_color="inverse"
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
    </div>
    """, unsafe_allow_html=True)
with tab4:
    st.header("High Risk Providers Overview")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        risk_threshold = st.slider("Minimum Risk Score", 0, 10, 3)
    with col2:
        limit = st.number_input("Max number of providers to show", 10, 100, 20)
    with col3:
        st.write("Include excluded Providers?")
        include_excluded = st.checkbox("""Yes""")
        st.checkbox("No")
    
    if st.button("Load High Risk Providers", key="high_risk"):
        with st.spinner("Loading data..."):
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
            WHERE total_risk_score >= {risk_threshold} AND currently_excluded <= {include_excluded} 
            ORDER BY total_risk_score DESC
            LIMIT {limit}
            """
            
            try:
                df = run_query_safe(query)
                # Display metrics
                st.dataframe(df)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Providers", len(df))
                with col2:
                    st.metric("Currently Excluded", df['currently_excluded'].sum())
                with col3:
                    avg_risk = df['total_risk_score'].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.2f}")
                with col4:
                    high_opioid = (df['opioid_rate'] > 0.3).sum() if 'opioid_rate' in df.columns else 0
                    st.metric("High Opioid Prescribers", high_opioid)
                
                # Display data
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "opioid_rate": st.column_config.ProgressColumn(
                            "Opioid Rate",
                            min_value=0,
                            max_value=1,
                        ),
                    }
                )
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


with tab3:
    st.header("Temporal Analysis")
    
    growth_type = st.selectbox(
        "Growth Trajectory",
        ["All", "Suspicious", "Growing", "Stable", "New_Provider"]
    )
    
    states = run_query_safe("SELECT DISTINCT STATE FROM CROSS_PROGRAM_RISK ORDER BY 1")['state']
    
    selected_state = st.selectbox(
        "Which State are you interested in?", 
        options=states,
        key="required_selectbox2",
        help="Please select a provider type to continue",
        index=None
    )
    
    
    if st.button("Load Temporal Analysis", key="temporal"):
        with st.spinner("Loading temporal analysis..."):
            where_clause = f"WHERE growth_trajectory = '{growth_type}'" if growth_type != "All" else ""
            
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
            
            try:
                df = run_query_safe(query)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
            
with tab2:
    with st.spinner("Fetching Data For Model..."):
        
        if 'cpr_table' not in st.session_state:
            st.session_state.cpr_table = run_query_safe("SELECT * FROM CROSS_PROGRAM_RISK")
        cross_prog_risk = st.session_state.cpr_table

        if 'ta_table' not in st.session_state:
            st.session_state.ta_table = run_query_safe("SELECT * FROM TEMPORAL_PEER_ANALYSIS")
        temporal_analysis = st.session_state.ta_table
                
        if 'ba_table' not in st.session_state:
            st.session_state.ba_table = run_query_safe("SELECT * FROM PROVIDER_BILLING_ANOMALIES")
        billing_anom = st.session_state.ba_table
    
        cross_prog_risk['year'] = pd.to_datetime(cross_prog_risk['year'])
        billing_anom['year'] = pd.to_datetime(billing_anom['year'])   
        temporal_analysis['exclusion_date'] = pd.to_datetime(temporal_analysis['exclusion_date'])
    
    local_cpr = rfm.LocalTable(cross_prog_risk, name = "cross_program_risk").infer_metadata()
    local_ta = rfm.LocalTable(temporal_analysis, name= "temporal_analysis").infer_metadata()
    local_ba = rfm.LocalTable(billing_anom, name= "billing_anomalies").infer_metadata()
    
    local_ta.primary_key = 'npi'
    local_ta['npi'].stype = 'ID'
    local_ta.time_column = None
    local_ta['temporal_risk_score'].stype = 'categorical'
    local_cpr['cross_program_risk_score'].stype = 'categorical'
    local_ba['risk_score'].stype ='categorical'

    
    graph = rfm.LocalGraph(tables=[
        local_ba,
        local_cpr,
        local_ta
    ])
    
    
    graph.link(src_table=local_cpr, fkey='npi', dst_table=local_ta)
    graph.link(src_table=local_ba, fkey='npi', dst_table=local_ta)

    
    with st.spinner("Model Loading..."):
        if 'model' not in st.session_state:
            st.session_state.model = rfm.KumoRFM(graph)
        model = st.session_state.model
        
    
    states = run_query_safe("SELECT DISTINCT STATE FROM CROSS_PROGRAM_RISK ORDER BY 1")['state']
    
    st.subheader("Select Filter Options")
    by_state = st.checkbox("By State")
    if by_state:
        selected_state = st.selectbox(
            "Which State are you investigating?", 
            options=states,
            key="required_selectbox",
            help="Please select a provider type to continue",
            index=None
        )
    by_npi = st.checkbox('By NPI')
    if by_npi:
        providers_avail = run_query_safe("SELECT NPI FROM PROVIDERS")
        specific_npi = st.text_input("Provide the NPI you are investigating: ")
        #if int(specific_npi) not in providers_avail['npi']:
            #st.warning("Not a valid Provider, predictions will not work")
        
    by_p_type = st.checkbox('By Provider Type')
    if by_p_type:
        p_types = run_query_safe("SELECT DISTINCT PROVIDER_TYPE FROM TEMPORAL_PEER_ANALYSIS")
        selected_p_type = st.selectbox("Select the Provider Type you are investigating: ", p_types, index = None)
        
    
    limit = st.number_input("Number of High Risk Providers to show", 1, 20, 5)
    
    col1, col2, col3 = st.columns([.25,.25,.5])
    
    with col1:
        cpr_button = st.button("Predict Cross Program Risk")
        
    with col2:
        br_button = st.button("Predict Billing Risk")
    
    if cpr_button:
        where_statement = "WHERE ID LIKE '%2023'"
        if by_state:
            where_statement = where_statement + f" AND STATE = '{selected_state}'"
        if by_npi:
            where_statement = where_statement + f" AND NPI = {int(specific_npi)}"
        if by_p_type:
            where_statement = where_statement + f" AND PART_B_SPECIALTY = '{selected_p_type}'"
        
        curr_vals = run_query_safe(f"""SELECT ID, npi, cross_program_risk_score 
                                   FROM CROSS_PROGRAM_RISK 
                                   {where_statement} 
                                   ORDER by cross_program_risk_score DESC 
                                   LIMIT {limit}""")

        for i in range(limit):
            with st.spinner(f"Predictions for {curr_vals['npi'][i]} loading..."):
                bad_docs = run_query_safe(f"SELECT * FROM CROSS_PROGRAM_RISK WHERE NPI = cast({curr_vals['npi'][i]} as int) order by 3 desc")


                risk_score_queries = {
                    'sub_query1' : f"PREDICT SUM(cross_program_risk.opioid_claim_rate_calc,0,12,months) > 0.3 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query2' : f"PREDICT SUM(cross_program_risk.long_acting_opioid_rate_calc,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query3' : f"PREDICT SUM(cross_program_risk.brand_preference_rate_calc,0,12,months) > 0.7 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query4' : f"PREDICT SUM(cross_program_risk.antipsychotic_claims_count,0,12,months) > 500 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query5' : f"PREDICT SUM(cross_program_risk.drug_to_medical_ratio_calc,0,12,months) > 5 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query6' : f"PREDICT SUM(cross_program_risk.directly_excluded,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query7' : f"PREDICT SUM(cross_program_risk.address_excluded_flag,0,12,months) = 1  FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                }
                
                query = f"PREDICT SUM(cross_program_risk.combined_intensity_score,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}"
                query2 = f"PREDICT SUM(cross_program_risk.opioid_claim_rate,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}"
                query3 = f"PREDICT SUM(cross_program_risk.opioid_patient_rate,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}"
                
                df2 = model.predict(query, num_hops=6)
                df3 = model.predict(query2, num_hops=6)
                df4 = model.predict(query3, num_hops=6)
                
                
                df = pd.DataFrame()
                for key in risk_score_queries:
                    sub_df = model.predict(risk_score_queries[key], num_hops=6)
                    sub_df['Query'] = risk_score_queries[key]
                    df = pd.concat([df,sub_df])
            
            st.header(f'High Cross Program Risk Provider: {bad_docs['npi'][0]}', divider=True)
            st.metric(value=df['TARGET_PRED'].sum(), label='Projected 2024 Cross Program Risk Score', border = True)
            st.dataframe(bad_docs)
            
            
            create_timeseries_chart(historical_df=bad_docs[['year','combined_intensity_score']], predicted_value=df2['TARGET_PRED'][0], target_column='combined_intensity_score')
            create_timeseries_chart(historical_df=bad_docs[['year','opioid_claim_rate']], predicted_value=df3['TARGET_PRED'][0], target_column='opioid_claim_rate')
            create_timeseries_chart(historical_df=bad_docs[['year','opioid_patient_rate']], predicted_value=df4['TARGET_PRED'][0], target_column='opioid_patient_rate')
            
            st.subheader("Raw Predictions for Score Creation")
            st.dataframe(df)
            
    if br_button:
        where_statement = "WHERE ID LIKE '%2023'"
        if by_state:
            where_statement = where_statement + f" AND STATE = '{selected_state}'"
        if by_npi:
            where_statement = where_statement + f" AND NPI = {int(specific_npi)}"
        if by_p_type:
            where_statement = where_statement + f" AND PROVIDER_TYPE = '{selected_p_type}'"
        
        curr_vals = run_query_safe(f"""SELECT ID, npi, risk_score 
                                   FROM PROVIDER_BILLING_ANOMALIES 
                                   {where_statement} 
                                   ORDER by risk_score DESC 
                                   LIMIT {limit}""")
        
        for i in range(limit):
            with st.spinner(f"Predictions for {curr_vals['npi'][i]} loading..."):
                bad_docs = run_query_safe(f"SELECT * FROM PROVIDER_BILLING_ANOMALIES WHERE NPI = cast({curr_vals['npi'][i]} as int) order by 3 desc")
                
                query = f'PREDICT SUM(billing_anomalies.total_medicare_reimbursement,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                query2 = f'PREDICT SUM(billing_anomalies.payment_per_service,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                query3 = f'PREDICT SUM(billing_anomalies.services_per_beneficiary,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                query4 = f'PREDICT SUM(billing_anomalies.part_d_claims,0,12,months) FOR temporal_analysis.npi = {curr_vals["npi"][i]}'
                risk_score_queries = {
                    'sub_query1' : f"PREDICT SUM(billing_anomalies.payment_zscore ,0,12,months) > 2 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query2' : f"PREDICT SUM(billing_anomalies.payment_growth_rate,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query3' : f"PREDICT SUM(billing_anomalies.service_growth_rate,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query4' : f"PREDICT SUM(billing_anomalies.opioid_prescribing_rate,0,12,months) > 0.3 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query5' : f"PREDICT SUM(billing_anomalies.opioid_prescriber_rate,0,12,months) > 0.5 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query6' : f"PREDICT SUM(billing_anomalies.charge_to_payment_ratio,0,12,months) > 3 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query7' : f"PREDICT SUM(billing_anomalies.oig_excluded_flag,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals["npi"][i]}",
                    'sub_query8' : f"PREDICT SUM(billing_anomalies.payment_outlier_flag,0,12,months) = 1 FOR temporal_analysis.npi = {curr_vals["npi"][i]}", 
                    'sub_query9' : f"PREDICT SUM(billing_anomalies.payment_zscore ,0,12,months) < 2 FOR temporal_analysis.npi = {curr_vals["npi"][i]}"      
                }

                df = model.predict(query, num_hops=6)
                df2 = model.predict(query2, num_hops=6)
                df3 = model.predict(query3, num_hops=6)
                df4 = model.predict(query4, num_hops=6)
                df5 = pd.DataFrame()
                for key in risk_score_queries:
                    sub_df = model.predict(risk_score_queries[key], num_hops=6)
                    sub_df['Query'] = risk_score_queries[key]
                    df5 = pd.concat([df5,sub_df])
            
            
            st.header(f'High Billing Risk Provider: {bad_docs['npi'][0]}', divider=True)
            st.metric(value=df5['TARGET_PRED'].sum(), label='Projected 2024 Billing Risk Score', border = True)
            st.dataframe(bad_docs)
            
            create_timeseries_chart(historical_df=bad_docs[['year','total_medicare_reimbursement']], predicted_value=df['TARGET_PRED'][0], target_column='total_medicare_reimbursement')
            create_timeseries_chart(historical_df=bad_docs[['year','payment_per_service']], predicted_value=df2['TARGET_PRED'][0], target_column='payment_per_service')
            create_timeseries_chart(historical_df=bad_docs[['year','services_per_beneficiary']], predicted_value=df3['TARGET_PRED'][0], target_column='services_per_beneficiary')
            create_timeseries_chart(historical_df=bad_docs[['year','part_d_claims']], predicted_value=df4['TARGET_PRED'][0], target_column='part_d_claims')
            st.subheader("Raw Predictions for Score Creation")
            st.dataframe(df5)
