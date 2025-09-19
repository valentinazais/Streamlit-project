import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Cross-Asset Market Regime Monitor", layout="wide")

def create_sample_data():
    """Generate sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2024-09-19', freq='D')
    
    # Sample price data
    np.random.seed(42)
    data = {}
    
    # Generate sample asset prices
    assets = ['S&P 500', 'Gold', 'Crude Oil', 'USD Index', 'VIX']
    base_prices = [4000, 2000, 80, 100, 20]
    
    for i, asset in enumerate(assets):
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_prices[i] * np.exp(np.cumsum(returns))
        data[asset] = pd.Series(prices, index=dates)
    
    return pd.DataFrame(data)

def create_performance_chart(data, selected_assets):
    """Create performance chart"""
    fig = go.Figure()
    
    for asset in selected_assets:
        if asset in data.columns:
            # Normalize to 100 at start
            normalized = (data[asset] / data[asset].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=data.index,
                y=normalized,
                name=asset,
                mode='lines'
            ))
    
    fig.update_layout(
        title="Asset Performance (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        height=500
    )
    
    return fig

def create_sample_heatmap():
    """Create sample performance heatmap"""
    assets = ['Equities', 'Gold', 'Oil', 'Bonds', 'USD']
    periods = ['1M', '3M', '6M', '1Y']
    
    # Sample performance data
    np.random.seed(42)
    performance = np.random.normal(2, 8, (len(assets), len(periods)))
    
    fig = go.Figure(data=go.Heatmap(
        z=performance,
        x=periods,
        y=assets,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{val:.1f}%" for val in row] for row in performance],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Asset Performance Heatmap (%)",
        height=400
    )
    
    return fig

def main():
    st.title("Cross-Asset Market Regime Monitor")
    st.markdown("*Demo version with sample data*")
    
    # Generate sample data
    data = create_sample_data()
    
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
            chart = create_performance_chart(filtered_data, selected_assets)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("Please select at least one asset")
    
    with col2:
        st.header("Latest Prices")
        if not filtered_data.empty:
            latest_prices = filtered_data.iloc[-1]
            for asset in selected_assets:
                if asset in latest_prices.index:
                    st.metric(
                        asset, 
                        f"${latest_prices[asset]:.2f}",
                        f"{np.random.normal(0, 2):.2f}%"
                    )
    
    # Term structure section (placeholder)
    st.header("Futures Term Structure")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Gold Futures")
        # Sample term structure
        contracts = ['Dec24', 'Mar25', 'Jun25', 'Sep25']
        prices = [2010, 2015, 2020, 2025]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=contracts, 
            y=prices, 
            mode='lines+markers',
            name='Gold'
        ))
        fig.update_layout(height=300, yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Oil Futures")
        oil_prices = [78, 79, 80, 81]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=contracts, 
            y=oil_prices, 
            mode='lines+markers',
            name='Oil',
            line=dict(color='orange')
        ))
        fig.update_layout(height=300, yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Yield curves
    st.header("Bond Yield Curves")
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("US Treasury")
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        us_yields = [5.2, 5.1, 4.8, 4.5, 4.2, 4.3, 4.5]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=maturities, 
            y=us_yields, 
            mode='lines+markers',
            name='US Treasury'
        ))
        fig.update_layout(height=300, yaxis_title="Yield (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col6:
        st.subheader("German Bunds")
        german_yields = [3.5, 3.4, 3.2, 2.8, 2.5, 2.4, 2.6]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=maturities, 
            y=german_yields, 
            mode='lines+markers',
            name='German Bunds',
            line=dict(color='red')
        ))
        fig.update_layout(height=300, yaxis_title="Yield (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance heatmap
    st.header("Asset Performance Comparison")
    heatmap = create_sample_heatmap()
    st.plotly_chart(heatmap, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Data updates daily. This is a demo version with simulated data.*")

if __name__ == "__main__":
    main()
