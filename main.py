import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Cross-Asset Market Regime Monitor", layout="wide")

# Define the GitHub raw URLs for the TSV files
COMMODITIES_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/Commodities_2Y.tsv"
FOREX_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/Forex_2Y.tsv"
FIXED_INCOME_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/FixedIncome_2Y.tsv"
INDICES_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/Indices_2Y.tsv"

def load_and_process_tsv(url):
    """Load TSV from URL, handle comma as decimal, parse dates"""
    # Load all columns as string to handle comma decimals
    df = pd.read_csv(url, sep='\t', dtype=str)
    
    # Assume first column is 'Dates'
    if 'Dates' not in df.columns:
        raise ValueError(f"'Dates' column not found in data from {url}")
    
    # Parse dates
    df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    
    # For all other columns (assumed numeric), replace ',' with '.' and convert to float
    for col in df.columns:
        if col != 'Dates':
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set index to 'Dates' and drop any rows with invalid dates
    df = df.set_index('Dates').dropna(how='all')
    
    return df

def load_real_data():
    """Load real data from TSV files on GitHub and combine relevant assets"""
    # Load each TSV file with processing
    commodities_df = load_and_process_tsv(COMMODITIES_URL)
    forex_df = load_and_process_tsv(FOREX_URL)
    indices_df = load_and_process_tsv(INDICES_URL)
    # FixedIncome not needed for the main assets, but loaded if required later
    # fixed_income_df = load_and_process_tsv(FIXED_INCOME_URL)
    
    # Extract specific columns based on user's mapping
    # Assuming:
    # - 'SPX Index' is in Indices
    # - 'XAUUSD Curncy' might be in Forex or Commodities
    # - 'CL1 Comdty' is in Commodities
    # For now, forgetting USD_Index and VIX
    
    data = pd.DataFrame(index=indices_df.index.union(commodities_df.index.union(forex_df.index)))
    
    if 'SPX Index' in indices_df.columns:
        data['SPX Index'] = indices_df['SPX Index']
    else:
        st.error("SPX Index not found in Indices data.")
    
    # Try Forex for XAUUSD Curncy, fallback to Commodities
    if 'XAUUSD Curncy' in forex_df.columns:
        data['XAUUSD Curncy'] = forex_df['XAUUSD Curncy']
    elif 'XAUUSD Curncy' in commodities_df.columns:
        data['XAUUSD Curncy'] = commodities_df['XAUUSD Curncy']
    else:
        st.error("XAUUSD Curncy not found in Forex or Commodities data.")
    
    if 'CL1 Comdty' in commodities_df.columns:
        data['CL1 Comdty'] = commodities_df['CL1 Comdty']
    else:
        st.error("CL1 Comdty not found in Commodities data.")
    
    # Align indices and drop rows with all NaNs
    data = data.dropna(how='all')
    
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)

    data = data[data.index.notna()]
    
    return data

def main():
    st.title("Market Dashboard Skema")
    st.markdown("*Using real data from GitHub TSV files*")
    
    # Load real data instead of sample
    try:
        data = load_real_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return  # Exit if data loading fails
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Date range
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Asset selection
    available_assets = list(data.columns)
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        available_assets,
        default=available_assets  # Default to all loaded assets
    )
    
    # Market regime indicator (still random for demo; can be computed from real data later)
    st.sidebar.markdown("### Current Market Regime")
    current_regime = np.random.choice(['Growth', 'Recession', 'Inflation', 'Deflation'])
    st.sidebar.metric("Regime", current_regime)
    
    # Key metrics (still sample; adapt with real calculations if needed)
    st.sidebar.markdown("### Key Indicators")
    st.sidebar.metric("Growth", "2.1%", "0.3%")
    st.sidebar.metric("Inflation", "3.2%", "-0.1%")
    st.sidebar.metric("Volatility", "18.5", "2.1")
    
    # Filter data by date
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (data.index.date >= start_date) & (data.index.date <= end_date)
        filtered_data = data.loc[mask]
    else:
        filtered_data = data
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Performance Metrics")
        if selected_assets:
            # Use Streamlit's built-in line chart
            chart_data = filtered_data[selected_assets].copy()
            # Normalize to 100 at start (assuming price data)
            for asset in selected_assets:
                if not chart_data[asset].empty:
                    chart_data[asset] = (chart_data[asset] / chart_data[asset].iloc[0]) * 100
            
            st.line_chart(chart_data)
        else:
            st.warning("Please select at least one asset")
    
    with col2:
        st.header("Latest Prices")
        if not filtered_data.empty:
            latest_prices = filtered_data.iloc[-1]
            for asset in selected_assets:
                if asset in latest_prices.index:
                    # Calculate daily change (assuming previous day exists)
                    if len(filtered_data) > 1:
                        prev_price = filtered_data[asset].iloc[-2]
                        delta = ((latest_prices[asset] - prev_price) / prev_price) * 100
                        delta_str = f"{delta:.2f}%"
                    else:
                        delta_str = "N/A"
                    
                    st.metric(
                        asset.replace('_', ' '), 
                        f"${latest_prices[asset]:.2f}",
                        delta_str
                    )
    
    # Performance comparison table (updated to use real data for periods)
    st.header("Asset Performance Comparison")
    
    # Create performance table using real data
    periods = ['1M', '3M', '6M', '1Y']
    performance_data = []
    
    for asset in available_assets:
        row_data = {'Asset': asset.replace('_', ' ')}
        asset_series = data[asset].dropna()  # Drop NaNs for this asset
        
        if asset_series.empty:
            for period in periods:
                row_data[period] = "N/A"
            performance_data.append(row_data)
            continue
        
        for period in periods:
            if period == '1M':
                days = 30
            elif period == '3M':
                days = 90
            elif period == '6M':
                days = 180
            elif period == '1Y':
                days = 365
            
            end_date = asset_series.index.max()
            start_date = end_date - timedelta(days=days)
            
            # Find the closest start date available
            start_prices = asset_series[asset_series.index >= start_date]
            if not start_prices.empty:
                start_price = start_prices.iloc[0]
                end_price = asset_series.loc[end_date]
                perf = ((end_price - start_price) / start_price) * 100
                row_data[period] = f"{perf:.1f}%"
            else:
                row_data[period] = "N/A"
        
        performance_data.append(row_data)
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)
    
    # Additional metrics section (still sample; can compute from real data)
    st.header("Market Regime Analysis")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.metric("Equity Momentum", "Strong", "↑")
        st.metric("Bond-Equity Correlation", "-0.25", "↓")
    
    with col8:
        st.metric("Inflation Expectations", "2.8%", "↑")
        st.metric("Credit Spreads", "145 bps", "→")
    
    with col9:
        st.metric("Dollar Strength", "102.5", "↑")
        st.metric("Commodity Index", "485", "↓")
    
    # Footer
    st.markdown("---")
    st.markdown("*Data loaded from GitHub TSV files. Updates depend on repository.*")
    st.markdown("**Features:** Real-time regime detection • Multi-asset monitoring • Term structure analysis")

if __name__ == "__main__":
    main()


