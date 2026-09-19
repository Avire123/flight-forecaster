import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from model_pipeline import generate_synthetic_dataset, TravelDealForecaster
from scraper import FlightScraper

st.set_page_config(
    page_title="Flight Fare Trend Forecaster",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data & models
@st.cache_data
def load_dataset():
    return generate_synthetic_dataset(3500)

@st.cache_resource
def get_trained_pipeline(df):
    forecaster = TravelDealForecaster()
    metrics = forecaster.train_with_time_split(df)
    return forecaster, metrics

# System Initialization
df_raw = load_dataset()
forecaster, metrics = get_trained_pipeline(df_raw)

# Sidebar Interface
st.sidebar.title("✈️ Flight Search Settings")
st.sidebar.markdown("Configure itinerary parameters or fetch live prices.")

origins = sorted(df_raw['origin'].unique())
dests = sorted(df_raw['destination'].unique())

preferred_origin = 'NBO' if 'NBO' in origins else origins[0]
preferred_destination = 'LHR' if 'LHR' in dests else ('DXB' if 'DXB' in dests else dests[1] if len(dests) > 1 else dests[0])

origin = st.sidebar.selectbox("Origin Airport", origins, index=origins.index(preferred_origin))
destination = st.sidebar.selectbox("Destination Airport", dests, index=dests.index(preferred_destination))

st.sidebar.caption("African carriers connecting to international destinations, priced in Kenyan shillings (KSh).")

if origin == destination:
    st.sidebar.error("Origin and Destination must be distinct.")

airlines = sorted(df_raw['airline'].unique())
african_airlines = ['Kenya Airways', 'Ethiopian Airlines', 'RwandAir', 'Air Tanzania', 'Airlink', 'South African Airways', 'FlySafair']
preferred_airline = next((code for code in african_airlines if code in airlines), airlines[0])
airline = st.sidebar.selectbox("Airline", airlines, index=airlines.index(preferred_airline))

days_until = st.sidebar.slider("Days Until Departure", min_value=1, max_value=60, value=14)
layovers = st.sidebar.selectbox("Layovers", [0, 1, 2], index=0)
current_price = st.sidebar.number_input("Current Ticket Price (KSh)", min_value=5000.0, max_value=500000.0, value=68000.0, step=1000.0)

# Live Scraper Trigger Option
st.sidebar.divider()
st.sidebar.subheader("🕷️ Live Web Scraper")
enable_live_scrape = st.sidebar.checkbox("Fetch Live Rates on Search", value=False)

if enable_live_scrape and st.sidebar.button("Run Playwright Scraper"):
    with st.spinner(f"Scraping dynamic fares for {origin} → {destination}..."):
        dep_date_str = (datetime.now() + timedelta(days=days_until)).strftime("%Y-%m-%d")
        scraper = FlightScraper(headless=True)
        live_results = asyncio.run(scraper.fetch_route_flights(origin, destination, dep_date_str))
        
        if live_results:
            st.sidebar.success(f"Extracted {len(live_results)} live listings!")
            st.sidebar.dataframe(pd.DataFrame(live_results)[['airline', 'price', 'layovers']].rename(columns={'price': 'price_ksh'}))
            current_price = float(live_results[0]['price'])
        else:
            st.sidebar.warning("No live listings scraped. Falling back to input values.")

# Prepare Search Record
dep_date = datetime.now() + timedelta(days=days_until)
search_date = datetime.now()

query_df = pd.DataFrame([{
    'search_date': search_date,
    'departure_date': dep_date,
    'origin': origin,
    'destination': destination,
    'airline': airline,
    'layovers': layovers,
    'days_until_flight': days_until,
    'current_price': current_price
}])

# Execute Predictions
prediction = forecaster.predict(query_df)
predicted_trend = prediction['trend']
confidence = prediction['confidence']
pred_future_price = prediction['predicted_future_price']
price_delta = pred_future_price - current_price

# Main Layout
st.title("✈️ African Flight & Global Travel Deal Forecaster")
st.markdown("Explore African airlines connecting to global hubs and compare fares in Kenyan shillings (KSh) to find the best booking window.")

# Section 1: Buy Now vs Wait Recommendation Widget
st.subheader("1. Recommendation & 7-Day Fare Forecast")

col_rec, col_future, col_conf = st.columns(3)

with col_rec:
    if predicted_trend == 'Rise':
        st.error("🚨 **RECOMMENDATION: BUY NOW**")
        st.write("Prices are predicted to **increase** over the next 7 days.")
    elif predicted_trend == 'Drop':
        st.success("⏳ **RECOMMENDATION: WAIT**")
        st.write("Prices are predicted to **drop** over the next 7 days.")
    else:
        st.info("⚖️ **RECOMMENDATION: STABLE**")
        st.write("Prices are expected to stay relatively flat.")

with col_future:
    st.metric(
        label="Predicted Price in 7 Days",
        value=f"KSh {pred_future_price:,.2f}",
        delta=f"{price_delta:+,.2f} KSh",
        delta_color="inverse"
    )

with col_conf:
    st.metric(
        label="Model Confidence Score",
        value=f"{confidence:.1f}%"
    )

st.divider()

# Section 2: Route Trend Explorer Visualizations
st.subheader("2. Interactive Route Trend Explorer")

# Filter Historical Data
route_history = df_raw[(df_raw['origin'] == origin) & (df_raw['destination'] == destination)].copy()

if route_history.empty:
    st.warning("No historical dataset entries for this exact route combination.")
else:
    tab_trajectory, tab_airlines, tab_table = st.tabs(["Fare Trajectory", "Airline Comparisons", "Historical Data View"])
    
    with tab_trajectory:
        st.markdown(f"**Price Decay Curve vs Days to Departure** (`{origin}` → `{destination}`)")
        
        fig_scatter = px.scatter(
            route_history,
            x='days_until_flight',
            y='current_price',
            color='airline',
            trendline="lowess",
            labels={'days_until_flight': 'Days Remaining to Flight', 'current_price': 'Price (KSh)'},
            template="plotly_white"
        )
        
        # Add current user query point
        fig_scatter.add_trace(
            go.Scatter(
                x=[days_until],
                y=[current_price],
                mode='markers+text',
                marker=dict(symbol='star', size=18, color='gold', line=dict(width=2, color='black')),
                name='Your Selected Fare',
                text=['Your Fare'],
                textposition='top center'
            )
        )

        fig_scatter.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_airlines:
        st.markdown("**Price Distribution Across Airlines**")
        fig_box = px.box(
            route_history,
            x='airline',
            y='current_price',
            color='airline',
            points="all",
            labels={'current_price': 'Ticket Fare (KSh)', 'airline': 'Carrier'},
            template="plotly_white"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with tab_table:
        st.markdown("**Historical Route Sample Data**")
        st.dataframe(route_history[['search_date', 'departure_date', 'airline', 'layovers', 'days_until_flight', 'current_price', 'price_trend_7d']].head(50))

# Section 3: Model Metrics & Architecture
with st.expander("📊 View Machine Learning Pipeline Metrics"):
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Classification Test Accuracy", f"{metrics['accuracy']:.2%}")
    m_col2.metric("Regression Test MAE", f"${metrics['mae']:.2f}")
    m_col3.metric("Validation Split Size", f"{metrics['test_samples']} samples")