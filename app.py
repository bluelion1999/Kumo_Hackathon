import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Medicare Fraud Detection", layout="wide")
st.title("Medicare Fraud Detection Dashboard")

# Initialize connection
@st.cache_resource
def init_connection():
    return st.connection("snowflake")

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
    
# Test connection
with st.spinner("Testing Snowflake connection..."):
    try:
        test_df = run_query_safe("SELECT CURRENT_VERSION() as VERSION, CURRENT_USER() as USER")
        st.success(f"✅ Connected as {test_df['USER'].iloc[0]}")
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")
        st.stop()

# Main application
tab1, tab2, tab3 = st.tabs(["High Risk Providers", "Billing Anomalies", "Temporal Analysis"])

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
                df
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

# Sidebar for additional options
with st.sidebar:
    st.header("Quick Actions")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    if st.button("Test Connection"):
        try:
            conn = init_connection()
            result = run_query_safe("SELECT 1")
            st.success("Connection successful!")
        except Exception as e:
            st.error(f"Connection failed: {e}")