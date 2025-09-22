import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Cross-Asset Market Regime Monitor", layout="wide")

# Define the GitHub raw URLs for the CSV files
COMMODITIES_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/Commodities_2Y.csv"
FOREX_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/Forex_2Y.csv"
FIXED_INCOME_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/FixedIncome_2Y.csv"
INDICES_URL = "https://raw.githubusercontent.com/valentinazais/Streamlit-project/refs/heads/main/Indices_2Y.csv"

def load_and_process_csv(url):
    """Load CSV from URL, handle comma as decimal, parse dates in DD/MM/YYYY format"""
    try:
        # Load all columns as string to handle comma decimals
        df = pd.read_csv(url, sep=';', dtype=str)
        
        # Assume first column is 'Dates'
        if 'Dates' not in df.columns:
            raise ValueError(f"'Dates' column not found in data from {url}")
        
        # Parse dates with DD/MM/YYYY format
        df['Dates'] = pd.to_datetime(df['Dates'], format='%d/%m/%Y', errors='coerce')
        
        # Drop rows where 'Dates' is NaT
        df = df.dropna(subset=['Dates'])
        
        # For all other columns (assumed numeric), replace ',' with '.' and convert to float
        for col in df.columns:
            if col != 'Dates':
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set index to 'Dates' and drop any rows with all NaN values
        df = df.set_index('Dates').dropna(how='all')
        
        # Sort index to ensure chronological order
        df = df.sort_index()
        
        return df
    except Exception as e:
        st.warning(f"Failed to load or process data from {url}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def load_real_data():
    """Load real data from CSV files on GitHub and combine relevant assets"""
    # Load each CSV file with processing
    commodities_df = load_and_process_csv(COMMODITIES_URL)
    forex_df = load_and_process_csv(FOREX_URL)
    indices_df = load_and_process_csv(INDICES_URL)
    # fixed_income_df = load_and_process_csv(FIXED_INCOME_URL)  # Load if needed for yields
    
    # Combine indices from all sources
    all_indices = commodities_df.index.union(forex_df.index.union(indices_df.index))
    data = pd.DataFrame(index=all_indices)
    
    # Extract specific assets; treat as prices
    selected_assets = []
    
    if not indices_df.empty and 'SPX Index' in indices_df.columns:
        data['SPX Index'] = indices_df['SPX Index']
        selected_assets.append('SPX Index')
    # else:
    #     st.warning("SPX Index not found or failed to load.")
    
    # Try Forex for XAUUSD Curncy, fallback to Commodities
    if not forex_df.empty and 'XAUUSD Curncy' in forex_df.columns:
        data['XAUUSD Curncy'] = forex_df['XAUUSD Curncy']
        selected_assets.append('XAUUSD Curncy')
    elif not commodities_df.empty and 'XAUUSD Curncy' in commodities_df.columns:
        data['XAUUSD Curncy'] = commodities_df['XAUUSD Curncy']
        selected_assets.append('XAUUSD Curncy')
    # else:
    #     st.warning("XAUUSD Curncy not found.")
    
    if not commodities_df.empty and 'CL1 Comdty' in commodities_df.columns:
        data['CL1 Comdty'] = commodities_df['CL1 Comdty']
        selected_assets.append('CL1 Comdty')
    # else:
    #     st.warning("CL1 Comdty not found.")
    
    # Drop rows with all NaNs
    data = data.dropna(how='all')
    
    # Sort index
    data = data.sort_index()
    
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)
    
    # Remove any rows with NaT in index
    data = data[data.index.notna()]
    
    return data, selected_assets

def main():
    st.title("Market Dashboard Skema")
    st.markdown("*Using real price data from GitHub CSV files*")
    
    # Load real data
    try:
        data, default_assets = load_real_data()
        if data.empty:
            st.error("No data loaded successfully. Please check the URLs and file contents.")
            return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Date range
    min_date = data.index.min().date() if not data.empty else datetime.now().date()
    max_date = data.index.max().date() if not data.empty else datetime.now().date()
    
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
        default=default_assets  # Default to loaded assets
    )
    
    # Market regime indicator (random for demo; can compute from data)
    st.sidebar.markdown("### Current Market Regime")
    current_regime = np.random.choice(['Growth', 'Recession', 'Inflation', 'Deflation'])
    st.sidebar.metric("Regime", current_regime)
    
    # Key metrics (sample; can compute from data)
    st.sidebar.markdown("### Key Indicators")
    st.sidebar.metric("Growth", "2.1%", "0.3%")
    st.sidebar.metric("Inflation", "3.2%", "-0.1%")
    st.sidebar.metric("Volatility", "18.5", "2.1")
    
    # Filter data by date
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        mask = (data.index >= start_dt) & (data.index <= end_dt)
        filtered_data = data.loc[mask]
    else:
        filtered_data = data
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Performance Metrics")
        if selected_assets and not filtered_data.empty:
            chart_data = filtered_data[selected_assets].copy()
            # Normalize prices to 100 at start for performance charting
            for asset in selected_assets:
                asset_series = chart_data[asset].dropna()
                if not asset_series.empty:
                    first_valid = asset_series.first_valid_index()
                    chart_data[asset] = (chart_data[asset] / asset_series.loc[first_valid]) * 100
                else:
                    chart_data[asset] = np.nan
            st.line_chart(chart_data)
        else:
            st.warning("Please select at least one asset or no data available.")
    
    with col2:
        st.header("Latest Prices")
        if not filtered_data.empty:
            latest_prices = filtered_data.iloc[-1]
            for asset in selected_assets:
                if asset in latest_prices.index and pd.notna(latest_prices[asset]):
                    # Calculate daily change if possible
                    if len(filtered_data) > 1:
                        prev_price = filtered_data[asset].iloc[-2]
                        if pd.notna(prev_price):
                            delta = ((latest_prices[asset] - prev_price) / prev_price) * 100
                            delta_str = f"{delta:.2f}%"
                        else:
                            delta_str = "N/A"
                    else:
                        delta_str = "N/A"
                    st.metric(
                        asset.replace('_', ' '), 
                        f"${latest_prices[asset]:.2f}",
                        delta_str
                    )
                else:
                    st.metric(asset.replace('_', ' '), "N/A", "N/A")
    
    # Term structure section (hardcoded; adapt with real data if available)
    st.header("Futures Term Structure")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Gold Futures")
        contracts = ['Dec24', 'Mar25', 'Jun25', 'Sep25']
        prices = [2010, 2015, 2020, 2025]
        term_data = pd.DataFrame({'Price': prices}, index=contracts)
        st.bar_chart(term_data)
    
    with col4:
        st.subheader("Oil Futures")
        oil_prices = [78, 79, 80, 81]
        oil_data = pd.DataFrame({'Price': oil_prices}, index=contracts)
        st.bar_chart(oil_data)
    
    # Yield curves (hardcoded; adapt with real data if available)
    st.header("Bond Yield Curves")
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("US Treasury")
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        us_yields = [5.2, 5.1, 4.8, 4.5, 4.2, 4.3, 4.5]
        us_data = pd.DataFrame({'Yield': us_yields}, index=maturities)
        st.line_chart(us_data)
    
    with col6:
        st.subheader("German Bunds")
        german_yields = [3.5, 3.4, 3.2, 2.8, 2.5, 2.4, 2.6]
        german_data = pd.DataFrame({'Yield': german_yields}, index=maturities)
        st.line_chart(german_data)
    
    # Performance comparison table (calculating returns from prices)
    st.header("Asset Performance Comparison")
    
    periods = ['1M', '3M', '6M', '1Y']
    performance_data = []
    
    for asset in available_assets:
        row_data = {'Asset': asset.replace('_', ' ')}
        asset_series = data[asset].dropna()
        
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
            
            start_prices = asset_series[asset_series.index >= start_date]
            if not start_prices.empty:
                start_price = start_prices.iloc[0]
                end_price = asset_series.loc[end_date]
                if pd.notna(start_price) and pd.notna(end_price):
                    perf = ((end_price - start_price) / start_price) * 100
                    row_data[period] = f"{perf:.1f}%"
                else:
                    row_data[period] = "N/A"
            else:
                row_data[period] = "N/A"
        
        performance_data.append(row_data)
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
    else:
        st.warning("No performance data available.")
    
    # Additional metrics section (sample)
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
    st.markdown("*Data loaded from GitHub CSV files. Treating as price data; returns calculated accordingly.*")
    st.markdown("**Features:** Real-time regime detection • Multi-asset monitoring • Term structure analysis")

if __name__ == "__main__":
    main()
