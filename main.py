import streamlit as st 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.title("Multi-Asset Financial Dashboard")

# Sidebar for time frame selection
st.sidebar.header("Time Frame Selection")
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Asset symbols
assets = {
    'Equity': {'SPY': 'S&P 500 ETF'},
    'Forex': {'DX-Y.NYB': 'USD Index'},
    'Commodities': {
        'GLD': 'Gold ETF',
        'USO': 'Crude Oil ETF',
        'WEAT': 'Wheat ETF'
    },
    'Bonds': {'SCHP': 'Inflation-Protected Securities ETF'}
}

# Function to get data and calculate metrics
@st.cache_data
def get_asset_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)['Close']
        returns = data.pct_change().dropna()
        cumulative_returns = (returns + 1).cumprod()
        
        # Calculate metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        total_return = (data.iloc[-1] / data.iloc[0] - 1) * 100
        
        return {
            'price': data,
            'returns': returns,
            'cumulative_returns': cumulative_returns,
            'volatility': volatility,
            'total_return': total_return
        }
    except:
        return None

# Get data for all assets
asset_data = {}
for category, symbols in assets.items():
    asset_data[category] = {}
    for symbol, name in symbols.items():
        data = get_asset_data(symbol, start_date, end_date)
        if data:
            asset_data[category][symbol] = {**data, 'name': name}

# Display cumulative returns chart
st.header("Cumulative Returns Comparison")
fig, ax = plt.subplots(figsize=(12, 8))

for category, symbols in asset_data.items():
    for symbol, data in symbols.items():
        if 'cumulative_returns' in data:
            ax.plot(data['cumulative_returns'].index, 
                   data['cumulative_returns'].values, 
                   label=f"{data['name']} ({symbol})")

ax.set_ylabel("Cumulative Returns")
ax.set_title("Multi-Asset Cumulative Returns")
ax.legend()
ax.set_yscale('log')
plt.xticks(rotation=45)
st.pyplot(fig)

# Display metrics table
st.header("Asset Metrics")
metrics_data = []
for category, symbols in asset_data.items():
    for symbol, data in symbols.items():
        if 'volatility' in data:
            metrics_data.append({
                'Asset': data['name'],
                'Symbol': symbol,
                'Category': category,
                'Total Return (%)': f"{data['total_return']:.2f}",
                'Volatility (%)': f"{data['volatility']*100:.2f}"
            })

if metrics_data:
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)

# Individual asset charts
st.header("Individual Asset Performance")
selected_category = st.selectbox("Select Category", list(assets.keys()))
selected_assets = st.multiselect(
    "Select Assets", 
    list(assets[selected_category].keys()),
    default=list(assets[selected_category].keys())
)

if selected_assets:
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol in selected_assets:
        if symbol in asset_data[selected_category]:
            data = asset_data[selected_category][symbol]
            ax.plot(data['price'].index, data['price'].values, 
                   label=f"{data['name']} ({symbol})")
    
    ax.set_ylabel("Price")
    ax.set_title(f"{selected_category} - Price History")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
