-- ============================================
-- Table 1: Provider Billing Anomaly Detection
-- ============================================

CREATE OR REPLACE TABLE MEDICARE_DATA.MODEL_READY.PROVIDER_BILLING_ANOMALIES AS
WITH provider_metrics AS (
    SELECT 
        CONCAT(CAST(b.npi as text) || cast(YEAR(b.YEAR) as text)) as ID,
        b.NPI,
        b."YEAR",
        b.PROVIDER_TYPE,
        b.STATE,
        CONCAT(b.STREET_ADDRESS_L1, ' ', b.CITY, ' ', b.STATE) AS provider_name_address,
        
        -- Basic metrics from Part B
        b.TOTAL_MEDICARE_REIMBURSEMENT,
        b.TOTAL_SERVICES,
        b.MEDICARE_BENEFICIARIES,
        b.UNIQUE_HCPCS_CODES,
        b.TOTAL_SUBMITTED_CHARGES,
        
        -- Calculated Part B metrics
        b.TOTAL_MEDICARE_REIMBURSEMENT / NULLIF(b.TOTAL_SERVICES, 0) AS payment_per_service,
        b.MEDICARE_BENEFICIARIES / NULLIF(b.TOTAL_SERVICES, 0) AS beneficiary_concentration,
        b.TOTAL_SERVICES / NULLIF(b.MEDICARE_BENEFICIARIES, 0) AS services_per_beneficiary,
        b.TOTAL_SUBMITTED_CHARGES / NULLIF(b.TOTAL_MEDICARE_REIMBURSEMENT, 0) AS charge_to_payment_ratio,
        
        -- Part D metrics (if provider also prescribes)
        d.TOTAL_CLAIMS AS part_d_claims,
        d.TOTAL_DRUG_COST,
        d.TOTAL_30DAY_FILLS,
        d.GENERIC_TOTAL_CLAIMS,
        d.BRAND_TOTAL_CLAIMS,
        d.OPIOID_TOTAL_CLAIMS,
        d.GENERIC_TOTAL_CLAIMS / NULLIF(d.TOTAL_CLAIMS, 0) AS generic_dispensing_rate,
        d.OPIOID_TOTAL_CLAIMS / NULLIF(d.TOTAL_CLAIMS, 0) AS opioid_prescribing_rate,
        d.OPIOID_PRESCRIBER_RATE,
        
        -- Risk factors from demographics
        b.PATIENT_AVG_RISK_SCORE,
        b.V1_PCT_ALC_DRUG,
        b.V1_PCT_ANXIETY,
        b.V2_PCT_ALZHEIMERS_DEMENTIA,
        
        -- OIG Exclusion Check
        CASE WHEN e.NPI IS NOT NULL THEN 1 ELSE 0 END AS oig_excluded_flag,
        e.EXCLUSION_DATE,
        e.EXCLUSION_TYPE
        
    FROM MEDICARE_DATA.MODEL_READY.PART_B b
    LEFT JOIN MEDICARE_DATA.MODEL_READY.PART_D d 
        ON b.NPI = d.NPI AND b."YEAR" = d."YEAR"
    LEFT JOIN MEDICARE_DATA.MODEL_READY.EXCLUSIONS e 
        ON b.NPI = e.NPI
),
peer_group_stats AS (
    SELECT 
        PROVIDER_TYPE,
        STATE,
        "YEAR",
        AVG(payment_per_service) AS avg_payment_per_service,
        STDDEV(payment_per_service) AS stddev_payment_per_service,
        AVG(services_per_beneficiary) AS avg_services_per_beneficiary,
        STDDEV(services_per_beneficiary) AS stddev_services_per_beneficiary,
        AVG(generic_dispensing_rate) AS avg_generic_rate,
        AVG(PATIENT_AVG_RISK_SCORE) AS avg_risk_score,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY TOTAL_MEDICARE_REIMBURSEMENT) AS p90_payment,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY TOTAL_MEDICARE_REIMBURSEMENT) AS p95_payment,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY TOTAL_MEDICARE_REIMBURSEMENT) AS p99_payment
    FROM provider_metrics
    WHERE PROVIDER_TYPE IS NOT NULL AND STATE IS NOT NULL
    GROUP BY PROVIDER_TYPE, STATE, "YEAR"
),
yoy_changes AS (
    SELECT 
        NPI,
        "YEAR",
        TOTAL_MEDICARE_REIMBURSEMENT,
        LAG(TOTAL_MEDICARE_REIMBURSEMENT) OVER (PARTITION BY NPI ORDER BY "YEAR") AS prev_year_payment,
        (TOTAL_MEDICARE_REIMBURSEMENT - LAG(TOTAL_MEDICARE_REIMBURSEMENT) OVER (PARTITION BY NPI ORDER BY "YEAR")) 
            / NULLIF(LAG(TOTAL_MEDICARE_REIMBURSEMENT) OVER (PARTITION BY NPI ORDER BY "YEAR"), 0) AS payment_growth_rate,
        LAG(TOTAL_SERVICES) OVER (PARTITION BY NPI ORDER BY "YEAR") AS prev_year_services,
        (TOTAL_SERVICES - LAG(TOTAL_SERVICES) OVER (PARTITION BY NPI ORDER BY "YEAR")) 
            / NULLIF(LAG(TOTAL_SERVICES) OVER (PARTITION BY NPI ORDER BY "YEAR"), 0) AS service_growth_rate
    FROM provider_metrics
)
SELECT 
    p.*,
    
    -- Add peer group statistics to the output table
    pg.avg_payment_per_service,
    pg.stddev_payment_per_service,
    pg.avg_services_per_beneficiary,
    pg.stddev_services_per_beneficiary,
    
    -- Peer comparison z-scores
    (p.payment_per_service - pg.avg_payment_per_service) / NULLIF(pg.stddev_payment_per_service, 0) AS payment_zscore,
    (p.services_per_beneficiary - pg.avg_services_per_beneficiary) / NULLIF(pg.stddev_services_per_beneficiary, 0) AS service_zscore,
    
    -- Year over year changes
    y.payment_growth_rate,
    y.service_growth_rate,
    CASE 
        WHEN y.payment_growth_rate > 0.5 THEN 1 
        ELSE 0 
    END AS suspicious_payment_growth,
    CASE 
        WHEN y.service_growth_rate > 0.5 THEN 1 
        ELSE 0 
    END AS suspicious_service_growth,
    
    -- Outlier flags
    CASE 
        WHEN p.TOTAL_MEDICARE_REIMBURSEMENT > pg.p95_payment THEN 1 
        ELSE 0 
    END AS payment_outlier_flag,
    CASE 
        WHEN p.TOTAL_MEDICARE_REIMBURSEMENT > pg.p99_payment THEN 1 
        ELSE 0 
    END AS extreme_payment_outlier,
    
    -- Generic prescribing anomaly (Part D)
    CASE 
        WHEN p.generic_dispensing_rate < (pg.avg_generic_rate - 0.2) 
            AND p.part_d_claims > 100 THEN 1 
        ELSE 0 
    END AS low_generic_flag,
    
    -- High risk patient concentration
    CASE 
        WHEN p.PATIENT_AVG_RISK_SCORE > (pg.avg_risk_score * 1.5) THEN 1 
        ELSE 0 
    END AS high_risk_patient_flag,
    
    -- Risk score calculation
    (CASE WHEN ABS((p.payment_per_service - pg.avg_payment_per_service) / NULLIF(pg.stddev_payment_per_service, 0)) > 2 THEN 1 ELSE 0 END +
     CASE WHEN y.payment_growth_rate > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN y.service_growth_rate > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN p.opioid_prescribing_rate > 0.3 THEN 1 ELSE 0 END +
     CASE WHEN p.OPIOID_PRESCRIBER_RATE > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN p.TOTAL_MEDICARE_REIMBURSEMENT > pg.p95_payment THEN 1 ELSE 0 END +
     CASE WHEN p.charge_to_payment_ratio > 3 THEN 1 ELSE 0 END +
     p.oig_excluded_flag) AS risk_score
     
FROM provider_metrics p
LEFT JOIN peer_group_stats pg 
    ON p.PROVIDER_TYPE = pg.PROVIDER_TYPE 
    AND p.STATE = pg.STATE 
    AND p."YEAR" = pg."YEAR"
LEFT JOIN yoy_changes y 
    ON p.NPI = y.NPI 
    AND p."YEAR" = y."YEAR";

-- ============================================
-- Table 2: Cross-Program Risk Indicators
-- ============================================

CREATE OR REPLACE TABLE MEDICARE_DATA.MODEL_READY.CROSS_PROGRAM_RISK AS
WITH dual_providers AS (
    SELECT DISTINCT
        CONCAT(CAST(b.npi as text) || cast(YEAR(b.YEAR) as text)) as ID,
        b.NPI,
        b."YEAR",
        b.STREET_ADDRESS_L1,
        b.CITY,
        b.STATE,
        b.ZIP,
        b.PROVIDER_TYPE AS part_b_specialty,
        d.PROVIDER_TYPE AS part_d_specialty,
        
        -- Part B metrics
        b.TOTAL_MEDICARE_REIMBURSEMENT AS part_b_payments,
        b.TOTAL_SERVICES AS part_b_services,
        b.MEDICARE_BENEFICIARIES AS part_b_beneficiaries,
        b.DRUG_MEDICARE_REIMBURSEMENT AS part_b_drug_payments,
        b.MED_MEDICARE_REIMBURSEMENT AS part_b_med_payments,
        
        -- Part D metrics  
        d.TOTAL_CLAIMS AS part_d_claims,
        d.TOTAL_DRUG_COST AS part_d_costs,
        d.TOTAL_BENEFICIARIES AS part_d_beneficiaries,
        d.OPIOID_TOTAL_CLAIMS,
        d.OPIOID_TOTAL_BENEFICIARIES,
        d.OPIOID_LA_TOTAL_CLAIMS,
        d.TOTAL_30DAY_FILLS,
        d.BRAND_TOTAL_CLAIMS,
        d.ANTIBIOTIC_TOTAL_CLAIMS,
        d.ANTIPSYCHOTIC_GE65_TOTAL_CLAIMS,
        
        -- Patient demographics alignment
        b.PATIENT_AVG_RISK_SCORE AS part_b_risk_score,
        d.PATIENT_AVG_RISK_SCORE AS part_d_risk_score,
        
        -- Calculate overlap potential
        LEAST(b.MEDICARE_BENEFICIARIES, d.TOTAL_BENEFICIARIES) AS max_possible_overlap
        
    FROM MEDICARE_DATA.MODEL_READY.PART_B b
    INNER JOIN MEDICARE_DATA.MODEL_READY.PART_D d 
        ON b.NPI = d.NPI AND b."YEAR" = d."YEAR"
),
address_exclusions AS (
    -- Check for providers at same address as excluded providers
    SELECT DISTINCT
        p.NPI,
        1 AS same_address_as_excluded
    FROM (
        SELECT NPI, STREET_ADDRESS_L1, CITY, STATE, ZIP 
        FROM MEDICARE_DATA.MODEL_READY.PART_B
        UNION
        SELECT NPI, STREET_ADDRESS_L1, CITY, STATE, ZIP 
        FROM MEDICARE_DATA.MODEL_READY.PART_D
    ) p
    INNER JOIN MEDICARE_DATA.MODEL_READY.EXCLUSIONS e 
        ON p.STREET_ADDRESS_L1 = e.ADDRESS 
        AND p.CITY = e.CITY
        AND p.STATE = e.STATE
    WHERE p.STREET_ADDRESS_L1 IS NOT NULL
)
SELECT 
    dp.*,
    
    -- Cross-program ratios
    dp.part_d_costs / NULLIF(dp.part_b_payments, 0) AS drug_to_medical_ratio,
    dp.part_b_drug_payments / NULLIF(dp.part_b_payments, 0) AS part_b_drug_ratio,
    dp.OPIOID_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) AS opioid_claim_rate,
    dp.OPIOID_TOTAL_BENEFICIARIES / NULLIF(dp.part_d_beneficiaries, 0) AS opioid_patient_rate,
    dp.OPIOID_LA_TOTAL_CLAIMS / NULLIF(dp.OPIOID_TOTAL_CLAIMS, 0) AS long_acting_opioid_rate,
    dp.TOTAL_30DAY_FILLS / NULLIF(dp.part_d_claims, 0) AS thirty_day_fill_rate,
    dp.BRAND_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) AS brand_preference_rate,
    dp.ANTIPSYCHOTIC_GE65_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) AS antipsychotic_rate,
    
    -- Beneficiary metrics
    dp.max_possible_overlap / NULLIF(GREATEST(dp.part_b_beneficiaries, dp.part_d_beneficiaries), 0) AS beneficiary_overlap_ratio,
    (dp.part_b_services / NULLIF(dp.part_b_beneficiaries, 0)) * 
    (dp.part_d_claims / NULLIF(dp.part_d_beneficiaries, 0)) AS combined_intensity_score,
    ABS(dp.part_b_risk_score - dp.part_d_risk_score) AS risk_score_divergence,
    
    -- Specialty mismatch detection
    CASE 
        WHEN UPPER(dp.part_b_specialty) LIKE '%RADIOLOGY%' AND dp.OPIOID_TOTAL_CLAIMS > 0 THEN 1
        WHEN UPPER(dp.part_b_specialty) LIKE '%PATHOLOGY%' AND dp.part_d_claims > 100 THEN 1
        WHEN UPPER(dp.part_b_specialty) LIKE '%EMERGENCY%' AND dp.part_d_claims > dp.part_b_services THEN 1
        WHEN UPPER(dp.part_b_specialty) LIKE '%ANESTHES%' AND dp.part_d_claims > 500 THEN 1
        ELSE 0
    END AS specialty_mismatch_flag,
    
    -- OIG association checks
    CASE WHEN e.NPI IS NOT NULL THEN 1 ELSE 0 END AS directly_excluded,
    COALESCE(ae.same_address_as_excluded, 0) AS address_excluded_flag,
    
    -- High risk patterns
    CASE 
        WHEN dp.OPIOID_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) > 0.4 
            AND UPPER(dp.part_b_specialty) LIKE '%PAIN%' THEN 1
        WHEN dp.OPIOID_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) > 0.3 
            AND dp.OPIOID_LA_TOTAL_CLAIMS > 100 THEN 1
        ELSE 0
    END AS pain_mill_risk_flag,
    
    -- Individual risk score components as separate columns (calculations, not flags)
    dp.OPIOID_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) AS opioid_claim_rate_calc,
    dp.OPIOID_LA_TOTAL_CLAIMS / NULLIF(dp.OPIOID_TOTAL_CLAIMS, 0) AS long_acting_opioid_rate_calc,
    dp.BRAND_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) AS brand_preference_rate_calc,
    dp.ANTIPSYCHOTIC_GE65_TOTAL_CLAIMS AS antipsychotic_claims_count,
    dp.part_d_costs / NULLIF(dp.part_b_payments, 0) AS drug_to_medical_ratio_calc,
    
    -- Risk score flags based on calculations
    CASE WHEN dp.OPIOID_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) > 0.3 THEN 1 ELSE 0 END AS high_opioid_rate_flag,
    CASE WHEN dp.OPIOID_LA_TOTAL_CLAIMS / NULLIF(dp.OPIOID_TOTAL_CLAIMS, 0) > 0.5 THEN 1 ELSE 0 END AS high_long_acting_opioid_flag,
    CASE WHEN dp.BRAND_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) > 0.7 THEN 1 ELSE 0 END AS high_brand_preference_flag,
    CASE WHEN dp.ANTIPSYCHOTIC_GE65_TOTAL_CLAIMS > 500 THEN 1 ELSE 0 END AS high_antipsychotic_flag,
    CASE WHEN e.NPI IS NOT NULL THEN 2 ELSE 0 END AS exclusion_points,
    CASE WHEN dp.part_d_costs / NULLIF(dp.part_b_payments, 0) > 5 THEN 1 ELSE 0 END AS high_drug_to_medical_ratio_flag,
    
    -- Calculate combined risk score using the individual components
    (CASE WHEN dp.OPIOID_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) > 0.3 THEN 1 ELSE 0 END +
     CASE WHEN dp.OPIOID_LA_TOTAL_CLAIMS / NULLIF(dp.OPIOID_TOTAL_CLAIMS, 0) > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN dp.BRAND_TOTAL_CLAIMS / NULLIF(dp.part_d_claims, 0) > 0.7 THEN 1 ELSE 0 END +
     CASE WHEN dp.ANTIPSYCHOTIC_GE65_TOTAL_CLAIMS > 500 THEN 1 ELSE 0 END +
     CASE WHEN e.NPI IS NOT NULL THEN 2 ELSE 0 END +
     COALESCE(ae.same_address_as_excluded, 0) +
     CASE WHEN dp.part_d_costs / NULLIF(dp.part_b_payments, 0) > 5 THEN 1 ELSE 0 END) AS cross_program_risk_score
     
FROM dual_providers dp
LEFT JOIN MEDICARE_DATA.MODEL_READY.EXCLUSIONS e ON dp.NPI = e.NPI
LEFT JOIN address_exclusions ae ON dp.NPI = ae.NPI;

-- ============================================
-- Table 3: Temporal and Peer Comparison Analysis
-- ============================================

CREATE OR REPLACE TABLE MEDICARE_DATA.MODEL_READY.TEMPORAL_PEER_ANALYSIS AS
WITH provider_timeline AS (
    SELECT 
        NPI,
        CONCAT(STREET_ADDRESS_L1, ' ', CITY, ' ', STATE) AS provider_name_address,
        PROVIDER_TYPE,
        STATE,
        "YEAR",
        TOTAL_MEDICARE_REIMBURSEMENT,
        TOTAL_SERVICES,
        MEDICARE_BENEFICIARIES,
        PATIENT_AVG_RISK_SCORE,
        MIN("YEAR") OVER (PARTITION BY NPI) AS first_year,
        MAX("YEAR") OVER (PARTITION BY NPI) AS last_year,
        COUNT(*) OVER (PARTITION BY NPI) AS years_active
    FROM MEDICARE_DATA.MODEL_READY.PART_B
),
three_year_trends AS (
    SELECT 
        NPI,
        MAX(provider_name_address) AS provider_name_address,
        MAX(PROVIDER_TYPE) AS PROVIDER_TYPE,
        MAX(STATE) AS STATE,
        
        -- Aggregate metrics across years
        AVG(TOTAL_MEDICARE_REIMBURSEMENT) AS avg_annual_payment,
        STDDEV(TOTAL_MEDICARE_REIMBURSEMENT) AS payment_variance,
        MAX(TOTAL_MEDICARE_REIMBURSEMENT) AS max_payment,
        MIN(TOTAL_MEDICARE_REIMBURSEMENT) AS min_payment,
        
        -- Extract year from DATE and calculate growth
        MAX(CASE WHEN YEAR( "YEAR") = 2023 THEN TOTAL_MEDICARE_REIMBURSEMENT END) AS payment_2023,
        MAX(CASE WHEN YEAR( "YEAR") = 2022 THEN TOTAL_MEDICARE_REIMBURSEMENT END) AS payment_2022,
        MAX(CASE WHEN YEAR( "YEAR") = 2021 THEN TOTAL_MEDICARE_REIMBURSEMENT END) AS payment_2021,
        
        -- Service trends
        MAX(CASE WHEN YEAR( "YEAR") = 2023 THEN TOTAL_SERVICES END) AS services_2023,
        MAX(CASE WHEN YEAR( "YEAR") = 2022 THEN TOTAL_SERVICES END) AS services_2022,
        MAX(CASE WHEN YEAR( "YEAR") = 2021 THEN TOTAL_SERVICES END) AS services_2021,
        
        -- Beneficiary trends
        MAX(CASE WHEN YEAR( "YEAR") = 2023 THEN MEDICARE_BENEFICIARIES END) AS beneficiaries_2023,
        MAX(CASE WHEN YEAR( "YEAR") = 2022 THEN MEDICARE_BENEFICIARIES END) AS beneficiaries_2022,
        MAX(CASE WHEN YEAR( "YEAR") = 2021 THEN MEDICARE_BENEFICIARIES END) AS beneficiaries_2021,
        
        -- Risk score trends
        AVG(PATIENT_AVG_RISK_SCORE) AS avg_patient_risk,
        
        -- Flag new entrants
        MIN(YEAR( "YEAR")) AS first_billing_year,
        COUNT(DISTINCT YEAR( "YEAR")) AS years_present
        
    FROM provider_timeline
    GROUP BY NPI
),
peer_percentiles AS (
    SELECT 
        PROVIDER_TYPE,
        STATE,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_annual_payment) AS p25_payment,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY avg_annual_payment) AS p50_payment,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_annual_payment) AS p75_payment,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY avg_annual_payment) AS p90_payment,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_annual_payment) AS p95_payment,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY avg_annual_payment) AS p99_payment
    FROM three_year_trends
    WHERE PROVIDER_TYPE IS NOT NULL AND STATE IS NOT NULL
    GROUP BY PROVIDER_TYPE, STATE
),
part_d_trends AS (
    SELECT 
        NPI,
        AVG(BRAND_TOTAL_CLAIMS / NULLIF(TOTAL_CLAIMS, 0)) AS avg_brand_rate,
        AVG(OPIOID_TOTAL_CLAIMS / NULLIF(TOTAL_CLAIMS, 0)) AS avg_opioid_rate,
        AVG(OPIOID_PRESCRIBER_RATE) AS avg_opioid_prescriber_rate,
        STDDEV(TOTAL_DRUG_COST) AS drug_cost_variance,
        MAX(CASE WHEN YEAR( "YEAR") = 2023 THEN TOTAL_DRUG_COST END) AS drug_cost_2023,
        MAX(CASE WHEN YEAR( "YEAR") = 2021 THEN TOTAL_DRUG_COST END) AS drug_cost_2021
    FROM MEDICARE_DATA.MODEL_READY.PART_D
    GROUP BY NPI
)
SELECT 
    t.*,
    
    -- Growth trajectory classification
    CASE 
        WHEN (t.payment_2023 - t.payment_2021) / NULLIF(t.payment_2021, 0) < 0.2 THEN 'Stable'
        WHEN (t.payment_2023 - t.payment_2021) / NULLIF(t.payment_2021, 0) BETWEEN 0.2 AND 0.5 THEN 'Growing'
        WHEN (t.payment_2023 - t.payment_2021) / NULLIF(t.payment_2021, 0) > 0.5 THEN 'Suspicious'
        WHEN t.payment_2021 IS NULL AND t.payment_2023 IS NOT NULL THEN 'New_Provider'
        ELSE 'Unknown'
    END AS growth_trajectory,
    
    -- Year over year growth rates
    (t.payment_2022 - t.payment_2021) / NULLIF(t.payment_2021, 0) AS growth_21_to_22,
    (t.payment_2023 - t.payment_2022) / NULLIF(t.payment_2022, 0) AS growth_22_to_23,
    
    -- Service growth rates
    (t.services_2023 - t.services_2021) / NULLIF(t.services_2021, 0) AS service_growth_21_to_23,
    
    -- Beneficiary growth
    (t.beneficiaries_2023 - t.beneficiaries_2021) / NULLIF(t.beneficiaries_2021, 0) AS beneficiary_growth,
    
    -- Consistency metric (coefficient of variation)
    t.payment_variance / NULLIF(t.avg_annual_payment, 0) AS payment_consistency,
    
    -- New provider flag
    CASE 
        WHEN t.first_billing_year >= 2022 AND t.payment_2023 > p.p75_payment THEN 1 
        ELSE 0 
    END AS suspicious_new_provider,
    
    -- Peer percentile rankings
    CASE 
        WHEN t.avg_annual_payment <= p.p25_payment THEN '0-25'
        WHEN t.avg_annual_payment <= p.p50_payment THEN '25-50'
        WHEN t.avg_annual_payment <= p.p75_payment THEN '50-75'
        WHEN t.avg_annual_payment <= p.p90_payment THEN '75-90'
        WHEN t.avg_annual_payment <= p.p95_payment THEN '90-95'
        WHEN t.avg_annual_payment <= p.p99_payment THEN '95-99'
        ELSE '99+'
    END AS peer_percentile_range,
    
    -- Outlier persistence (count of years in top 5%)
    (CASE WHEN t.payment_2021 > p.p95_payment THEN 1 ELSE 0 END +
     CASE WHEN t.payment_2022 > p.p95_payment THEN 1 ELSE 0 END +
     CASE WHEN t.payment_2023 > p.p95_payment THEN 1 ELSE 0 END) AS years_as_outlier,
    
    -- Part D metrics if applicable
    d.avg_brand_rate,
    d.avg_opioid_rate,
    d.avg_opioid_prescriber_rate,
    (d.drug_cost_2023 - d.drug_cost_2021) / NULLIF(d.drug_cost_2021, 0) AS drug_cost_growth,
    
    -- OIG exclusion timing
    e.EXCLUSION_DATE,
    e.EXCLUSION_TYPE,
    CASE 
        WHEN e.NPI IS NOT NULL AND e.EXCLUSION_DATE >= '2021-01-01' THEN 'Excluded_During_Period'
        WHEN e.NPI IS NOT NULL AND e.EXCLUSION_DATE < '2021-01-01' THEN 'Previously_Excluded'
        ELSE 'Not_Excluded'
    END AS exclusion_status,
    
    -- Final temporal risk score
    (CASE WHEN (t.payment_2023 - t.payment_2021) / NULLIF(t.payment_2021, 0) > 0.5 THEN 2 ELSE 0 END +
     CASE WHEN (t.services_2023 - t.services_2021) / NULLIF(t.services_2021, 0) > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN t.first_billing_year >= 2022 AND t.payment_2023 > p.p75_payment THEN 1 ELSE 0 END +
     CASE WHEN t.payment_variance / NULLIF(t.avg_annual_payment, 0) > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN t.avg_annual_payment > p.p95_payment THEN 1 ELSE 0 END +
     CASE WHEN d.avg_opioid_prescriber_rate > 0.5 THEN 1 ELSE 0 END +
     CASE WHEN e.NPI IS NOT NULL THEN 2 ELSE 0 END) AS temporal_risk_score
     
FROM three_year_trends t
LEFT JOIN peer_percentiles p 
    ON t.PROVIDER_TYPE = p.PROVIDER_TYPE 
    AND t.STATE = p.STATE
LEFT JOIN part_d_trends d 
    ON t.NPI = d.NPI
LEFT JOIN MEDICARE_DATA.MODEL_READY.EXCLUSIONS e 
    ON t.NPI = e.NPI;
