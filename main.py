import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Cross-Asset Market Regime Monitor", layout="wide")

def create_sample_data():
    """Generate sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2024-09-19', freq='D')
    
    # Sample price data
    np.random.seed(68)
    data = {}
    
    # Generate sample asset prices
    assets = ['S&P_500', 'Gold', 'Crude_Oil', 'USD_Index', 'VIX']
    base_prices = [4000, 2000, 80, 100, 20]
    
    for i, asset in enumerate(assets):
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_prices[i] * np.exp(np.cumsum(returns))
        data[asset] = prices
    
    return pd.DataFrame(data, index=dates)

def main():
    st.title("Market Dashboard Skema")
    st.markdown("*Demo version with sample data*")
    
    # Generate sample data
    data = create_sample_data()
    
    # Sidebar
    st.sidebar.header("Filrzeazerzer")
    
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
        default=available_assets[:3]
    )
    
    # Market regime indicator
    st.sidebar.markdown("### Current Market Regime")
    current_regime = np.random.choice(['Growth', 'Recession', 'Inflation', 'Deflation'])
    st.sidebar.metric("Regime", current_regime)
    
    # Key metrics
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
            # Normalize to 100 at start
            for asset in selected_assets:
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
                    st.metric(
                        asset.replace('_', ' '), 
                        f"${latest_prices[asset]:.2f}",
                        f"{np.random.normal(0, 2):.2f}%"
                    )
    
    # Term structure section
    st.header("Futures Term Structure")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Gold Futures")
        contracts = ['Dec24', 'Mar25', 'Jun25', 'Sep25']
        prices = [2010, 2015, 2020, 2025]
        
        term_data = pd.DataFrame({
            'Price': prices
        }, index=contracts)
        st.bar_chart(term_data)
    
    with col4:
        st.subheader("Oil Futures")
        oil_prices = [78, 79, 80, 81]
        
        oil_data = pd.DataFrame({
            'Price': oil_prices
        }, index=contracts)
        st.bar_chart(oil_data)
    
    # Yield curves
    st.header("Bond Yield Curves")
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("US Treasury")
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        us_yields = [5.2, 5.1, 4.8, 4.5, 4.2, 4.3, 4.5]
        
        us_data = pd.DataFrame({
            'Yield': us_yields
        }, index=maturities)
        st.line_chart(us_data)
    
    with col6:
        st.subheader("German Bunds")
        german_yields = [3.5, 3.4, 3.2, 2.8, 2.5, 2.4, 2.6]
        
        german_data = pd.DataFrame({
            'Yield': german_yields
        }, index=maturities)
        st.line_chart(german_data)
    
    # Performance comparison table
    st.header("Asset Performance Comparison")
    
    # Create performance table
    periods = ['1M', '3M', '6M', '1Y']
    performance_data = []
    
    for asset in available_assets:
        row_data = {'Asset': asset.replace('_', ' ')}
        for period in periods:
            perf = np.random.normal(2, 8)
            row_data[period] = f"{perf:.1f}%"
        performance_data.append(row_data)
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)
    
    # Additional metrics section
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
    st.markdown("*Data updates daily. This is a demo version with simulated data.*")
    st.markdown("**Features:** Real-time regime detection • Multi-asset monitoring • Term structure analysis")

if __name__ == "__main__":
    main()






