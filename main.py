import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import altair as alt

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
        'INDU Index', 'NDX Index', 'SPX Index', 'CAC Index', 'TPX Index'
    ],
    'Fixed Income': [
        'GB3 Govt', 'GB6 Govt', 'GB12 Govt', 'GT2 Govt', 'GT5 Govt', 'GT10 Govt', 'GT30 Govt'
    ]
}

# All fixed income tickers are US Treasuries/Bills
US_FIXED_INCOME_TICKERS = TICKERS['Fixed Income']

# Flatten all tickers for easy access
ALL_TICKERS = [ticker for group in TICKERS.values() for ticker in group]

# Region-specific groupings (based on user request; note: TPX500 not in current data, using available)
FOREX_REGIONS = {
    'USA': 'USDCAD Curncy',  # USD-CAD for USA
    'EMEA': 'EURUSD Curncy',  # EUR-USD for EMEA
    'Asia': 'USDJPY Curncy'   # USD-JPY for Asia
}

INDICES_REGIONS = {
    'US': 'SPX Index',    # SPX for US
    'EMEA': 'CAC Index',  # CAC for EMEA
    'Asia': 'TPX500 Index'   # Placeholder; TPX500 not available, using NDX (adjust CSV if needed)
}

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

def get_yield_curve_data(data, tickers, yc_datetime):
    # Find the closest date in the index (nearest previous if not exact)
    if yc_datetime in data.index:
        selected_yields = data.loc[yc_datetime, tickers].dropna()
    else:
        # Get the nearest previous date
        prev_dates = data.index[data.index <= yc_datetime]
        if not prev_dates.empty:
            nearest_date = prev_dates.max()
            selected_yields = data.loc[nearest_date, tickers].dropna()
        else:
            selected_yields = pd.Series()
    
    if not selected_yields.empty:
        # Extract maturities (e.g., '3M' from 'GB3 Govt', '2Y' from 'GT2 Govt')
        maturities = []
        months_list = []  # New: track months for explicit sorting
        for t in selected_yields.index:
            # Use regex to extract the numeric part cleanly (removes 'GB', 'GT', ' Govt', and trims)
            maturity_num = re.sub(r'GB|GT| Govt', '', t).strip()
            if maturity_num.isdigit():  # Ensure it's a number
                num = int(maturity_num)
                if t.startswith('GB'):
                    if maturity_num == '12':
                        maturities.append('1Y')
                        months_list.append(12)  # 1Y = 12 months
                    else:
                        maturities.append(maturity_num + 'M')
                        months_list.append(num)  # e.g., 3M = 3 months
                elif t.startswith('GT'):
                    maturities.append(maturity_num + 'Y')
                    months_list.append(num * 12)  # e.g., 2Y = 24 months
                else:
                    maturities.append(t)  # Fallback
                    months_list.append(0)
            else:
                maturities.append(t)  # Fallback if parsing fails
                months_list.append(0)
        
        yc_data = pd.DataFrame({
            'Maturity': maturities,
            'Yield': selected_yields.values,
            'Months': months_list  # New column for sorting
        })
        
        # Sort by 'Months' to ensure order from shortest to longest maturity
        yc_data = yc_data.sort_values(by='Months').reset_index(drop=True)
        
        return yc_data
    return pd.DataFrame()

def compute_market_regime(data):
    """Compute market regime based on yield curve inversion (simple logic)"""
    latest_date = data.index.max()
    yc_data = get_yield_curve_data(data, US_FIXED_INCOME_TICKERS, latest_date)
    if not yc_data.empty:
        short_yield = yc_data[yc_data['Maturity'] == '2Y']['Yield'].values[0] if '2Y' in yc_data['Maturity'].values else np.nan
        long_yield = yc_data[yc_data['Maturity'] == '10Y']['Yield'].values[0] if '10Y' in yc_data['Maturity'].values else np.nan
        if not np.isnan(short_yield) and not np.isnan(long_yield):
            if short_yield > long_yield:
                return 'Recession'  # Inverted curve
            elif long_yield - short_yield > 1:
                return 'Growth'
            else:
                return 'Stable'
    return 'Unknown'

def compute_key_indicators(data):
    """Compute real key indicators from data"""
    spx_ticker = 'SPX Index' if 'SPX Index' in data.columns else None
    oil_ticker = 'CL1 Comdty' if 'CL1 Comdty' in data.columns else None
    
    indicators = {}
    
    # Growth: 1Y return of SPX
    if spx_ticker:
        spx_series = data[spx_ticker].dropna()
        if len(spx_series) > 365:
            end_price = spx_series.iloc[-1]
            start_price = spx_series.iloc[-366]  # Approx 1Y back
            growth = ((end_price - start_price) / start_price) * 100
            growth_delta = growth - (((spx_series.iloc[-31] - spx_series.iloc[-366]) / spx_series.iloc[-366]) * 100)  # Delta vs 1M ago
            indicators['Growth'] = (f"{growth:.1f}%", f"{growth_delta:.1f}%")
        else:
            indicators['Growth'] = ("N/A", "N/A")
    else:
        indicators['Growth'] = ("N/A", "N/A")
    
    # Inflation: 1Y change in oil price as proxy
    if oil_ticker:
        oil_series = data[oil_ticker].dropna()
        if len(oil_series) > 365:
            end_price = oil_series.iloc[-1]
            start_price = oil_series.iloc[-366]
            inflation = ((end_price - start_price) / start_price) * 100
            inflation_delta = inflation - (((oil_series.iloc[-31] - oil_series.iloc[-366]) / oil_series.iloc[-366]) * 100)
            indicators['Inflation'] = (f"{inflation:.1f}%", f"{inflation_delta:.1f}%")
        else:
            indicators['Inflation'] = ("N/A", "N/A")
    else:
        indicators['Inflation'] = ("N/A", "N/A")
    
    # Volatility: 30-day rolling std dev of SPX returns
    if spx_ticker:
        returns = compute_returns(data[[spx_ticker]])
        vol = returns.rolling(window=30).std().iloc[-1][spx_ticker] * np.sqrt(252) * 100  # Annualized
        prev_vol = returns.rolling(window=30).std().iloc[-2][spx_ticker] * np.sqrt(252) * 100
        vol_delta = vol - prev_vol
        indicators['Volatility'] = (f"{vol:.1f}", f"{vol_delta:.1f}")
    else:
        indicators['Volatility'] = ("N/A", "N/A")
    
    return indicators

def compute_regime_analysis(data):
    """Compute real regime analysis metrics from data"""
    spx_ticker = 'SPX Index' if 'SPX Index' in data.columns else None
    bond_ticker = 'GT10 Govt' if 'GT10 Govt' in data.columns else None
    oil_ticker = 'CL1 Comdty' if 'CL1 Comdty' in data.columns else None
    usd_ticker = 'EURUSD Curncy' if 'EURUSD Curncy' in data.columns else None
    comm_index_ticker = 'GC1 Comdty' if 'GC1 Comdty' in data.columns else None  # Gold as proxy
    
    analysis = {}
    
    # Equity Momentum: 1M return of SPX
    if spx_ticker:
        spx_series = data[spx_ticker].dropna()
        if len(spx_series) > 30:
            end = spx_series.iloc[-1]
            start = spx_series.iloc[-31]
            momentum = ((end - start) / start) * 100
            analysis['Equity Momentum'] = (f"{momentum:.1f}%", "↑" if momentum > 0 else "↓")
        else:
            analysis['Equity Momentum'] = ("N/A", "→")
    else:
        analysis['Equity Momentum'] = ("N/A", "→")
    
    # Bond-Equity Correlation: 90-day corr between bond and SPX
    if spx_ticker and bond_ticker:
        returns = compute_returns(data[[spx_ticker, bond_ticker]])
        corr = returns.rolling(90).corr().iloc[-1][spx_ticker]
        prev_corr = returns.rolling(90).corr().iloc[-2][spx_ticker]
        delta = "↓" if corr < prev_corr else "↑" if corr > prev_corr else "→"
        analysis['Bond-Equity Correlation'] = (f"{corr:.2f}", delta)
    else:
        analysis['Bond-Equity Correlation'] = ("N/A", "→")
    
    # Inflation Expectations: 1M change in oil
    if oil_ticker:
        oil_series = data[oil_ticker].dropna()
        if len(oil_series) > 30:
            end = oil_series.iloc[-1]
            start = oil_series.iloc[-31]
            infl_exp = ((end - start) / start) * 100
            analysis['Inflation Expectations'] = (f"{infl_exp:.1f}%", "↑" if infl_exp > 0 else "↓")
        else:
            analysis['Inflation Expectations'] = ("N/A", "→")
    else:
        analysis['Inflation Expectations'] = ("N/A", "→")
    
    # Credit Spreads: Placeholder (use 10Y - 2Y spread as proxy)
    latest_date = data.index.max()
    yc_data = get_yield_curve_data(data, US_FIXED_INCOME_TICKERS, latest_date)
    if not yc_data.empty:
        short = yc_data[yc_data['Maturity'] == '2Y']['Yield'].values[0] if '2Y' in yc_data['Maturity'].values else np.nan
        long = yc_data[yc_data['Maturity'] == '10Y']['Yield'].values[0] if '10Y' in yc_data['Maturity'].values else np.nan
        if not np.isnan(short) and not np.isnan(long):
            spread = (long - short) * 100  # in bps
            analysis['Credit Spreads'] = (f"{spread:.0f} bps", "→")  # Placeholder delta
        else:
            analysis['Credit Spreads'] = ("N/A", "→")
    else:
        analysis['Credit Spreads'] = ("N/A", "→")
    
    # Dollar Strength: Latest EURUSD (inverse as dollar strength)
    if usd_ticker:
        usd = data[usd_ticker].iloc[-1]
        prev_usd = data[usd_ticker].iloc[-2]
        delta = "↑" if usd < prev_usd else "↓"  # Lower EURUSD means stronger USD
        analysis['Dollar Strength'] = (f"{1/usd:.2f}", delta)  # Inverse for strength
    else:
        analysis['Dollar Strength'] = ("N/A", "→")
    
    # Commodity Index: Gold as proxy
    if comm_index_ticker:
        comm = data[comm_index_ticker].iloc[-1]
        prev_comm = data[comm_index_ticker].iloc[-2]
        delta = "↑" if comm > prev_comm else "↓"
        analysis['Commodity Index'] = (f"{comm:.0f}", delta)
    else:
        analysis['Commodity Index'] = ("N/A", "→")
    
    return analysis

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
    
    # Compute real metrics for sidebar
    current_regime = compute_market_regime(data)
    key_indicators = compute_key_indicators(data)
    
    # Market regime indicator
    st.sidebar.markdown("### Current Market Regime")
    st.sidebar.metric("Regime", current_regime)
    
    # Key metrics
    st.sidebar.markdown("### Key Indicators")
    growth_val, growth_delta = key_indicators.get('Growth', ("N/A", "N/A"))
    st.sidebar.metric("Growth", growth_val, growth_delta)
    infl_val, infl_delta = key_indicators.get('Inflation', ("N/A", "N/A"))
    st.sidebar.metric("Inflation", infl_val, infl_delta)
    vol_val, vol_delta = key_indicators.get('Volatility', ("N/A", "N/A"))
    st.sidebar.metric("Volatility", vol_val, vol_delta)
    
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
    
    # Yield curves section
    st.header("Bond Yield Curves")
    
    # Date selector for yield curve
    yc_date = st.date_input(
        "Select Date for Yield Curve",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="yc_date"
    )
    yc_datetime = datetime.combine(yc_date, datetime.min.time())
    
    # US Treasury Yields (including all fixed income as US)
    st.subheader("US Treasury Yield Curve")
    if US_FIXED_INCOME_TICKERS and any(t in data.columns for t in US_FIXED_INCOME_TICKERS):
        us_yc_data = get_yield_curve_data(data, US_FIXED_INCOME_TICKERS, yc_datetime)
        if not us_yc_data.empty:
            # Calculate dynamic y-axis domain with padding for auto-zoom
            min_yield = us_yc_data['Yield'].min()
            max_yield = us_yc_data['Yield'].max()
            padding = 0.5  # Adjust this value (e.g., 0.2 for tighter zoom, 1.0 for more space)
            y_domain = [min_yield - padding, max_yield + padding]
            
            # Custom Altair chart for explicit x-order (shortest to longest) and auto-zoomed y-axis
            yc_chart = alt.Chart(us_yc_data).mark_line(point=True).encode(
                x=alt.X('Maturity:N', sort=None, title='Maturity'),  # Use 'Maturity' column, respect DataFrame order
                y=alt.Y('Yield:Q', title='Yield (%)', scale=alt.Scale(domain=y_domain)),  # Dynamic y-scale
                tooltip=['Maturity', 'Yield']
            ).properties(
                width=700,
                height=400,
                title='US Treasury Yield Curve'
            ).interactive()
            
            st.altair_chart(yc_chart, use_container_width=True)
        else:
            st.warning(f"No US Treasury yield data available for {yc_date} or nearest previous date.")
    else:
        st.warning("No US Treasury yield data available.")
    
    # Forex Dashboard
    st.header("Forex Dashboard")
    forex_period = st.selectbox("Select Period for Forex", ['1M', '3M', '6M', '1Y', 'All'], index=4)
    
    if forex_period != 'All':
        if forex_period == '1M':
            days = 30
        elif forex_period == '3M':
            days = 90
        elif forex_period == '6M':
            days = 180
        elif forex_period == '1Y':
            days = 365
        forex_end = data.index.max()
        forex_start = forex_end - timedelta(days=days)
        forex_data = data.loc[(data.index >= forex_start) & (data.index <= forex_end)]
    else:
        forex_data = data
    
    forex_cols = st.columns(3)
    for i, (region, ticker) in enumerate(FOREX_REGIONS.items()):
        with forex_cols[i]:
            st.subheader(f"{region} ({ticker})")
            if ticker in forex_data.columns and not forex_data[ticker].dropna().empty:
                # Prepare data for Altair (reset index for 'Date')
                chart_data = forex_data[[ticker]].reset_index().melt('Dates', var_name='Ticker', value_name='Rate')
                chart_data = chart_data.dropna(subset=['Rate'])
                
                # Calculate dynamic y-domain with padding
                min_rate = chart_data['Rate'].min()
                max_rate = chart_data['Rate'].max()
                padding = (max_rate - min_rate) * 0.05  # 5% padding relative to range
                if padding == 0: padding = 0.01  # Small padding if flat
                y_domain = [min_rate - padding, max_rate + padding]
                
                # Altair chart with auto-fit y-axis
                forex_chart = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X('Dates:T', title='Date'),
                    y=alt.Y('Rate:Q', title='Rate', scale=alt.Scale(domain=y_domain)),
                    tooltip=['Dates', 'Rate']
                ).properties(
                    height=300,
                    title=f"{ticker} Rate"
                ).interactive()
                
                st.altair_chart(forex_chart, use_container_width=True)
                
                latest = forex_data[ticker].iloc[-1]
                if len(forex_data) > 1:
                    prev = forex_data[ticker].iloc[-2]
                    delta = ((latest - prev) / prev) * 100
                    st.metric(f"Latest Rate for {ticker}", f"{latest:.4f}", f"{delta:.2f}%")
                else:
                    st.metric(f"Latest Rate for {ticker}", f"{latest:.4f}", "N/A")
            else:
                st.warning(f"No data for {ticker}")
    
    # Indices Dashboard
    st.header("Indices Dashboard")
    indices_period = st.selectbox("Select Period for Indices", ['1M', '3M', '6M', '1Y', 'All'], index=4)
    
    if indices_period != 'All':
        if indices_period == '1M':
            days = 30
        elif indices_period == '3M':
            days = 90
        elif indices_period == '6M':
            days = 180
        elif indices_period == '1Y':
            days = 365
        indices_end = data.index.max()
        indices_start = indices_end - timedelta(days=days)
        indices_data = data.loc[(data.index >= indices_start) & (data.index <= indices_end)]
    else:
        indices_data = data
    
    indices_cols = st.columns(3)
    for i, (region, ticker) in enumerate(INDICES_REGIONS.items()):
        with indices_cols[i]:
            st.subheader(f"{region} ({ticker})")
            if ticker in indices_data.columns and not indices_data[ticker].dropna().empty:
                # Normalize for performance
                chart_data = indices_data[[ticker]].copy()
                first_valid = chart_data[ticker].first_valid_index()
                if first_valid is not None:
                    chart_data[ticker] = (chart_data[ticker] / chart_data[ticker].loc[first_valid]) * 100
                
                # Prepare data for Altair (reset index for 'Date')
                chart_data = chart_data.reset_index().melt('Dates', var_name='Ticker', value_name='Normalized Level')
                chart_data = chart_data.dropna(subset=['Normalized Level'])
                
                # Calculate dynamic y-domain with padding
                min_level = chart_data['Normalized Level'].min()
                max_level = chart_data['Normalized Level'].max()
                padding = (max_level - min_level) * 0.05  # 5% padding relative to range
                if padding == 0: padding = 1  # Small padding if flat
                y_domain = [min_level - padding, max_level + padding]
                
                # Altair chart with auto-fit y-axis
                indices_chart = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X('Dates:T', title='Date'),
                    y=alt.Y('Normalized Level:Q', title='Normalized Level (100 at start)', scale=alt.Scale(domain=y_domain)),
                    tooltip=['Dates', 'Normalized Level']
                ).properties(
                    height=300,
                    title=f"{ticker} Performance"
                ).interactive()
                
                st.altair_chart(indices_chart, use_container_width=True)
                
                latest = indices_data[ticker].iloc[-1]
                if len(indices_data) > 1:
                    prev = indices_data[ticker].iloc[-2]
                    delta = ((latest - prev) / prev) * 100
                    st.metric(f"Latest Level for {region} ({ticker})", f"{latest:.2f}", f"{delta:.2f}%")
                else:
                    st.metric(f"Latest Level for {region} ({ticker})", f"{latest:.2f}", "N/A")
            else:
                st.warning(f"No data for {ticker}")
    
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
    
    # Correlation Matrix
    st.header("Correlation Matrix")
    if selected_assets and len(selected_assets) >= 2 and not filtered_data.empty:
        returns = compute_returns(filtered_data[selected_assets])
        corr_matrix = compute_correlation_matrix(returns)
        
        # Display as styled dataframe without background_gradient (to avoid matplotlib dependency)
        styled_corr = corr_matrix.style.format("{:.2f}")
        st.dataframe(styled_corr, use_container_width=True)
    else:
        st.warning("Select at least two assets to compute correlation matrix.")
    
    # Rolling Correlations
    st.header("Rolling Correlations")
    if available_assets:
        ticker1 = st.selectbox("Select First Ticker", available_assets)
        ticker2 = st.selectbox("Select Second Ticker", available_assets)
        
        if ticker1 and ticker2 and ticker1 != ticker2:
            returns = compute_returns(filtered_data[[ticker1, ticker2]])
            
            windows = {
                '1W (7 days)': 7,
                '1M (30 days)': 30,
                '3M (90 days)': 90
            }
            
            selected_window = st.selectbox("Select Rolling Window", list(windows.keys()))
            window_size = windows[selected_window]
            
            rolling_corr = compute_rolling_correlation(returns, ticker1, ticker2, window_size)
            
            st.line_chart(rolling_corr)
        else:
            st.warning("Select two different tickers to compute rolling correlation.")
    else:
        st.warning("No assets available for correlation analysis.")
    
    # Market Regime Analysis (computed from real data)
    st.header("Market Regime Analysis")
    regime_analysis = compute_regime_analysis(data)
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        eq_mom_val, eq_mom_delta = regime_analysis.get('Equity Momentum', ("N/A", "→"))
        st.metric("Equity Momentum", eq_mom_val, eq_mom_delta)
        bond_eq_corr_val, bond_eq_corr_delta = regime_analysis.get('Bond-Equity Correlation', ("N/A", "→"))
        st.metric("Bond-Equity Correlation", bond_eq_corr_val, bond_eq_corr_delta)
    
    with col8:
        infl_exp_val, infl_exp_delta = regime_analysis.get('Inflation Expectations', ("N/A", "→"))
        st.metric("Inflation Expectations", infl_exp_val, infl_exp_delta)
        credit_spread_val, credit_spread_delta = regime_analysis.get('Credit Spreads', ("N/A", "→"))
        st.metric("Credit Spreads", credit_spread_val, credit_spread_delta)
    
    with col9:
        dollar_str_val, dollar_str_delta = regime_analysis.get('Dollar Strength', ("N/A", "→"))
        st.metric("Dollar Strength", dollar_str_val, dollar_str_delta)
        comm_index_val, comm_index_delta = regime_analysis.get('Commodity Index', ("N/A", "→"))
        st.metric("Commodity Index", comm_index_val, comm_index_delta)
    
    # Footer
    st.markdown("---")
    st.markdown("*Data loaded from GitHub CSV files. All hardcoded data replaced with CSV-loaded tickers.*")
    st.markdown("**Features:** Real-time regime detection • Multi-asset monitoring • Correlation analysis")

if __name__ == "__main__":
    main()




