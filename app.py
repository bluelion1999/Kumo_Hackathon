import streamlit as st
import pandas as pd
import time
import kumoai.experimental.rfm as rfm
import os

st.set_page_config(page_title="Medicare Fraud Detection", layout="wide")
st.title("Medicare Fraud Detection Dashboard")

# Initialize connection
@st.cache_resource
def init_connection():
    return st.connection("snowflake")

rfm.init(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1ZDJkOTI3YmVhNDIwMjAzODdhNDc1YzRkZWYxNWM1ZSIsImp0aSI6IjU3MTdhYzc5LTE5OGMtNGRiYS04YjQzLTM5OTc0M2Y3NDk0ZCIsImlhdCI6MTc1NDc3NDE4NywiZXhwIjoxNzU5OTU4MTg3fQ.So8CMCTwgRF0frTrv-d0v4AE41fKwclrvaIMz-WIxNY')

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

# Sidebar for additional options
with st.sidebar:
    st.header("Quick Actions")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")


# Main application
tab1, tab2, tab3, tab4= st.tabs(["High Risk Providers", "Billing Anomalies", "Temporal Analysis","Kumo_playground"])

with tab1:
    st.header("High Risk Providers Overview")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        risk_threshold = st.slider("Minimum Risk Score", 0, 10, 3)
    with col2:
        limit = st.number_input("Number of providers to show", 10, 100, 20)
    
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
                opioid_rate
            FROM MEDICARE_DATA.MODEL_READY.HIGH_RISK_PROVIDERS
            WHERE total_risk_score >= {risk_threshold}
            ORDER BY total_risk_score DESC
            LIMIT {limit}
            """
            
            try:
                df = run_query_safe(query)
                # Display metrics
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
                        "total_risk_score": st.column_config.ProgressColumn(
                            "Risk Score",
                            min_value=0,
                            max_value=10,
                        ),
                        "opioid_rate": st.column_config.ProgressColumn(
                            "Opioid Rate",
                            min_value=0,
                            max_value=1,
                        ),
                    }
                )
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

with tab2:
    st.header("Billing Anomalies Analysis")
    
    if st.button("Load Billing Anomalies", key="billing"):
        with st.spinner("Loading billing anomalies..."):
            query = """
            SELECT 
                NPI,
                provider_name_address,
                STATE,
                YEAR,
                payment_zscore,
                payment_growth_rate,
                suspicious_payment_growth,
                opioid_prescribing_rate,
                risk_score
            FROM MEDICARE_DATA.MODEL_READY.PROVIDER_BILLING_ANOMALIES
            WHERE risk_score >= 3
                AND YEAR(YEAR) = 2023
            ORDER BY risk_score DESC
            LIMIT 50
            """
            
            try:
                df = run_query_safe(query)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Temporal Analysis")
    
    growth_type = st.selectbox(
        "Growth Trajectory",
        ["All", "Suspicious", "Growing", "Stable", "New_Provider"]
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
            
            
with tab4:
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
    
    selected_state = st.selectbox("Which State are you interested in?", states)
    
    
    st.write(selected_state)
    
    if st.button("Predict Temporal Risk"):
        curr_vals = run_query_safe("SELECT npi, temporal_risk_score FROM TEMPORAL_PEER_ANALYSIS ORDER by 2 DESC LIMIT 25")
        out_table = pd.DataFrame()

        for i in range(25):
            query = f'PREDICT SUM(temporal_analysis.temporal_risk_score, 0,360, days) > {curr_vals['temporal_risk_score'][i]} FOR temporal_analysis.npi = {curr_vals['npi'][i]}'
            df = model.predict(query)
            out_table = pd.concat([out_table, df])
        
        st.dataframe(out_table)
        
        
    if st.button("Predict Cross Program Risk"):
        st.write()
    if st.button("Predict Billing Risk"):
        curr_vals = run_query_safe("SELECT ID, risk_score FROM PROVIDER_BILLING_ANOMALIES WHERE ID LIKE '%2023' ORDER by 2 DESC LIMIT 25")

        query = 'PREDICT billing_anomalies.risk_score FOR billing_anomalies.id in ({providers})'
        df = model.predict(query.format(providers=', '.join(f"'{x}'" for x in curr_vals['id'])))
        df = df.merge(curr_vals, how='inner', left_on= 'ENTITY', right_on='id')

        st.dataframe(df)

    
    
    
