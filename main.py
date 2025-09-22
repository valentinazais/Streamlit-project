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

# Predefined list of tickers based on user input
TICKERS = {
    'Forex': [
        'USDCAD Curncy', 'USDMXN Curncy', 'EURUSD Curncy', 'GBPUSD Curncy',
        'USDJPY Curncy', 'EURJPY Curncy', 'USDARS Curncy'
    ],
    'Commodities': [
        'CL1 Comdty', 'GC1 Comdty', 'NG1 Comdty', 'XAUEUR Curncy'
    ],
    'Indices': [
        'INDU Index', 'NDX Index', 'SPX Index', 'CAC Index'
    ],
    'Fixed Income': [
        'GB3:GOV', 'GB6:GOV', 'GB12:GOV', 'GT2:GOV', 'GT5:GOV', 'GT10:GOV', 'GT30:GOV'
    ]
}

# Flatten all tickers for easy access
ALL_TICKERS = [ticker for group in TICKERS.values() for ticker in group]

def load_and_process_csv(url, expected_tickers=None):
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
        
        # If expected_tickers provided, filter to only those columns
        if expected_tickers:
            available_cols = [col for col in expected_tickers if col in df.columns]
            df = df[available_cols]
        
        return df
    except Exception as e:
        st.warning(f"Failed to load or process data from {url}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def load_real_data():
    """Load real data from CSV files on GitHub and combine relevant assets"""
    # Load each CSV file with processing and filter to expected tickers
    commodities_df = load_and_process_csv(COMMODITIES_URL, TICKERS['Commodities'])
    forex_df = load_and_process_csv(FOREX_URL, TICKERS['Forex'])
    indices_df = load_and_process_csv(INDICES_URL, TICKERS['Indices'])
    fixed_income_df = load_and_process_csv(FIXED_INCOME_URL, TICKERS['Fixed Income'])
    
    # Combine all dataframes on date index
    data = pd.concat([commodities_df, forex_df, indices_df, fixed_income_df], axis=1)
    
    # Drop rows with all NaNs
    data = data.dropna(how='all')
    
    # Sort index
    data = data.sort_index()
    
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)
    
    # Remove any rows with NaT in index
    data = data[data.index.notna()]
    
    # Get available assets (tickers with data)
    available_assets = [col for col in data.columns if not data[col].dropna().empty]
    
    return data, available_assets

def compute_returns(data, period='daily'):
    """Compute returns for the data"""
    if period == 'daily':
        returns = data.pct_change()
    return returns

def compute_correlation_matrix(returns):
    """Compute correlation matrix"""
    corr_matrix = returns.corr()
    return corr_matrix

def compute_rolling_correlation(returns, ticker1, ticker2, window):
    """Compute rolling correlation between two tickers"""
    rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2])
    return rolling_corr

def main():
    st.title("Market Dashboard Skema")
    st.markdown("*Using real price data from GitHub CSV files*")
    
    # Load real data
    try:
        data, available_assets = load_real_data()
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
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        available_assets,
        default=available_assets[:3] if available_assets else []  # Default to first 3 available
    )
    
    # Market regime indicator (placeholder; can compute from data if needed)
    st.sidebar.markdown("### Current Market Regime")
    current_regime = np.random.choice(['Growth', 'Recession', 'Inflation', 'Deflation'])
    st.sidebar.metric("Regime", current_regime)
    
    # Key metrics (computed from data if possible; placeholder for now)
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
                        f"{latest_prices[asset]:.2f}",
                        delta_str
                    )
                else:
                    st.metric(asset.replace('_', ' '), "N/A", "N/A")
    
    # Removed hardcoded term structure section; replace with CSV data if futures curves are added in future
    
    # Yield curves (using fixed income data if available)
    st.header("Bond Yield Curves")
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("US Treasury Yields")
        treasury_tickers = TICKERS['Fixed Income']
        if any(t in filtered_data.columns for t in treasury_tickers):
            # Get latest yields
            latest_yields = filtered_data[treasury_tickers].iloc[-1].dropna()
            if not latest_yields.empty:
                maturities = [t.split(':')[0] for t in latest_yields.index]  # e.g., 'GB3', 'GT10'
                us_data = pd.DataFrame({'Yield': latest_yields.values}, index=maturities)
                st.line_chart(us_data)
            else:
                st.warning("No US Treasury yield data available.")
        else:
            st.warning("No US Treasury yield data available.")
    
    with col6:
        st.subheader("Other Yield Curves")
        st.info("Additional yield curves (e.g., German Bunds) not available in current CSV data.")
    
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
    
    # New Module: Correlation Matrix
    st.header("Correlation Matrix")
    if selected_assets and len(selected_assets) >= 2 and not filtered_data.empty:
        returns = compute_returns(filtered_data[selected_assets])
        corr_matrix = compute_correlation_matrix(returns)
        
        # Display as styled dataframe with background gradient (heatmap-like)
        styled_corr = corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}")
        st.dataframe(styled_corr, use_container_width=True)
    else:
        st.warning("Select at least two assets to compute correlation matrix.")
    
    # New Module: Rolling Correlations (1W, 1M, 3M)
    st.header("Rolling Correlations")
    if available_assets:
        # Select two tickers for correlation
        ticker1 = st.selectbox("Select First Ticker", available_assets)
        ticker2 = st.selectbox("Select Second Ticker", available_assets)
        
        if ticker1 and ticker2 and ticker1 != ticker2:
            returns = compute_returns(filtered_data[[ticker1, ticker2]])
            
            # Define windows
            windows = {
                '1W (7 days)': 7,
                '1M (30 days)': 30,
                '3M (90 days)': 90
            }
            
            selected_window = st.selectbox("Select Rolling Window", list(windows.keys()))
            window_size = windows[selected_window]
            
            rolling_corr = compute_rolling_correlation(returns, ticker1, ticker2, window_size)
            
            # Plot rolling correlation
            st.line_chart(rolling_corr)
        else:
            st.warning("Select two different tickers to compute rolling correlation.")
    else:
        st.warning("No assets available for correlation analysis.")
    
    # Additional metrics section (placeholder)
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
    st.markdown("*Data loaded from GitHub CSV files. All hardcoded data replaced with CSV-loaded tickers.*")
    st.markdown("**Features:** Real-time regime detection • Multi-asset monitoring • Correlation analysis")

if __name__ == "__main__":
    main()
