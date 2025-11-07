"""
Streamlit Dashboard for NYC Airbnb Analysis

This interactive dashboard allows users to explore the NYC Airbnb dataset
with filters and visualizations for market analysis and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os

# Add src directory to path for imports
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="NYC Airbnb Market Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    try:
        df = pd.read_csv('outputs/cleaned_airbnb.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please run the analysis pipeline first.")
        return None

@st.cache_data
def load_models():
    """Load trained models."""
    try:
        rf_model = joblib.load('outputs/model_price_rf.joblib')
        return {'random_forest': rf_model}
    except FileNotFoundError:
        st.warning("Models not found. Prediction features will be disabled.")
        return None

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 NYC Airbnb Market Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    models = load_models()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Borough filter
    boroughs = ['All'] + sorted(df['neighbourhood_group'].unique().tolist())
    selected_borough = st.sidebar.selectbox("Select Borough", boroughs)
    
    # Room type filter
    room_types = ['All'] + sorted(df['room_type'].unique().tolist())
    selected_room_type = st.sidebar.selectbox("Select Room Type", room_types)
    
    # Price range filter
    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    price_range = st.sidebar.slider(
        "Price Range ($)", 
        price_min, price_max, 
        (price_min, price_max)
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_borough != 'All':
        filtered_df = filtered_df[filtered_df['neighbourhood_group'] == selected_borough]
    if selected_room_type != 'All':
        filtered_df = filtered_df[filtered_df['room_type'] == selected_room_type]
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Geographic Analysis", "💰 Price Analysis", "🤖 Predictions"])
    
    with tab1:
        overview_tab(filtered_df, df)
    
    with tab2:
        geographic_tab(filtered_df)
    
    with tab3:
        price_analysis_tab(filtered_df)
    
    with tab4:
        if models:
            prediction_tab(filtered_df, models)
        else:
            st.warning("Prediction models not available. Please run the modeling pipeline first.")

def overview_tab(filtered_df, full_df):
    """Overview tab with key metrics and summary."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Listings", 
            f"{len(filtered_df):,}",
            f"{len(filtered_df) - len(full_df):,}" if len(filtered_df) != len(full_df) else None
        )
    
    with col2:
        avg_price = filtered_df['price'].mean()
        full_avg_price = full_df['price'].mean()
        st.metric(
            "Average Price", 
            f"${avg_price:.0f}",
            f"${avg_price - full_avg_price:+.0f}" if len(filtered_df) != len(full_df) else None
        )
    
    with col3:
        median_price = filtered_df['price'].median()
        full_median_price = full_df['price'].median()
        st.metric(
            "Median Price", 
            f"${median_price:.0f}",
            f"${median_price - full_median_price:+.0f}" if len(filtered_df) != len(full_df) else None
        )
    
    with col4:
        avg_availability = filtered_df['availability_365'].mean()
        full_avg_availability = full_df['availability_365'].mean()
        st.metric(
            "Avg Availability", 
            f"{avg_availability:.0f} days",
            f"{avg_availability - full_avg_availability:+.0f}" if len(filtered_df) != len(full_df) else None
        )
    
    # Market share by borough
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Share by Borough")
        borough_counts = filtered_df['neighbourhood_group'].value_counts()
        fig_pie = px.pie(
            values=borough_counts.values, 
            names=borough_counts.index,
            title="Listings Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Room Type Distribution")
        room_counts = filtered_df['room_type'].value_counts()
        fig_bar = px.bar(
            x=room_counts.index, 
            y=room_counts.values,
            title="Listings by Room Type"
        )
        fig_bar.update_layout(xaxis_title="Room Type", yaxis_title="Number of Listings")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Key insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Key Insights")
    
    top_borough = filtered_df['neighbourhood_group'].value_counts().index[0]
    top_room_type = filtered_df['room_type'].value_counts().index[0]
    price_range_info = f"${filtered_df['price'].min():.0f} - ${filtered_df['price'].max():.0f}"
    
    st.write(f"• **Dominant Borough**: {top_borough} leads with {filtered_df['neighbourhood_group'].value_counts().iloc[0]:,} listings")
    st.write(f"• **Popular Room Type**: {top_room_type} represents {filtered_df['room_type'].value_counts().iloc[0]/len(filtered_df)*100:.1f}% of listings")
    st.write(f"• **Price Range**: {price_range_info} with ${filtered_df['price'].std():.0f} standard deviation")
    st.write(f"• **Market Activity**: Average {filtered_df['number_of_reviews'].mean():.0f} reviews per listing")
    st.markdown('</div>', unsafe_allow_html=True)

def geographic_tab(filtered_df):
    """Geographic analysis tab."""
    
    st.subheader("🗺️ Geographic Distribution")
    
    # Map visualization
    if len(filtered_df) > 0:
        # Add slider to control number of listings shown on map
        max_listings_on_map = st.slider(
            "Number of listings to show on map",
            min_value=1000,
            max_value=min(50000, len(filtered_df)),
            value=min(10000, len(filtered_df)),
            step=1000,
            help="Showing too many listings may slow down the map. Adjust for performance."
        )
        
        # Sample data for performance if needed
        if len(filtered_df) > max_listings_on_map:
            map_df = filtered_df.sample(n=max_listings_on_map, random_state=42)
        else:
            map_df = filtered_df
        
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="price",
            size="number_of_reviews",
            hover_data=["neighbourhood_group", "room_type", "price"],
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=10,
            title=f"Airbnb Listings Map (showing {len(map_df):,} of {len(filtered_df):,} listings)"
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Borough comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Borough")
        borough_prices = filtered_df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
        fig_borough = px.bar(
            x=borough_prices.index,
            y=borough_prices.values,
            title="Price Comparison Across Boroughs"
        )
        fig_borough.update_layout(xaxis_title="Borough", yaxis_title="Average Price ($)")
        st.plotly_chart(fig_borough, use_container_width=True)
    
    with col2:
        st.subheader("Availability by Borough")
        borough_availability = filtered_df.groupby('neighbourhood_group')['availability_365'].mean().sort_values(ascending=False)
        fig_avail = px.bar(
            x=borough_availability.index,
            y=borough_availability.values,
            title="Average Availability by Borough"
        )
        fig_avail.update_layout(xaxis_title="Borough", yaxis_title="Days Available per Year")
        st.plotly_chart(fig_avail, use_container_width=True)

def price_analysis_tab(filtered_df):
    """Price analysis tab."""
    
    st.subheader("💰 Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig_hist = px.histogram(
            filtered_df, 
            x="price", 
            nbins=50,
            title="Price Distribution"
        )
        fig_hist.add_vline(x=filtered_df['price'].mean(), line_dash="dash", 
                          annotation_text=f"Mean: ${filtered_df['price'].mean():.0f}")
        fig_hist.add_vline(x=filtered_df['price'].median(), line_dash="dot", 
                          annotation_text=f"Median: ${filtered_df['price'].median():.0f}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Reviews")
        fig_scatter = px.scatter(
            filtered_df.sample(n=min(1000, len(filtered_df))),
            x="number_of_reviews",
            y="price",
            color="room_type",
            title="Price vs Number of Reviews"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Price by room type
    st.subheader("Price Analysis by Room Type")
    fig_box = px.box(
        filtered_df,
        x="room_type",
        y="price",
        title="Price Distribution by Room Type"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Price statistics table
    st.subheader("Price Statistics by Borough and Room Type")
    price_stats = filtered_df.groupby(['neighbourhood_group', 'room_type'])['price'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    price_stats.columns = ['Count', 'Mean ($)', 'Median ($)', 'Std Dev ($)']
    st.dataframe(price_stats, use_container_width=True)

def prediction_tab(filtered_df, models):
    """Price prediction tab."""
    
    st.subheader("🤖 Price Prediction Tool")
    
    st.write("Use our trained Random Forest model to predict Airbnb prices based on property characteristics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Input fields for prediction
        pred_borough = st.selectbox("Borough", filtered_df['neighbourhood_group'].unique())
        pred_room_type = st.selectbox("Room Type", filtered_df['room_type'].unique())
        pred_min_nights = st.number_input("Minimum Nights", min_value=1, max_value=30, value=3)
        pred_availability = st.slider("Availability (days/year)", 0, 365, 180)
        pred_reviews = st.number_input("Number of Reviews", min_value=0, max_value=500, value=10)
        pred_host_listings = st.number_input("Host Total Listings", min_value=1, max_value=50, value=1)
    
    with col2:
        st.subheader("Location (Optional)")
        
        # Get average coordinates for selected borough
        borough_coords = filtered_df[filtered_df['neighbourhood_group'] == pred_borough][['latitude', 'longitude']].mean()
        
        pred_lat = st.number_input("Latitude", 
                                  value=float(borough_coords['latitude']), 
                                  format="%.6f")
        pred_lon = st.number_input("Longitude", 
                                  value=float(borough_coords['longitude']), 
                                  format="%.6f")
        
        if st.button("🎯 Predict Price", type="primary"):
            # Prepare features for prediction (simplified version)
            # Note: This is a simplified prediction - full implementation would require
            # proper feature engineering matching the training pipeline
            
            try:
                # Create feature vector (simplified)
                features = np.array([[
                    pred_lat, pred_lon, pred_min_nights, pred_reviews,
                    0.0,  # reviews_per_month (simplified)
                    pred_host_listings, pred_availability,
                    365 - pred_availability,  # active_days
                    0.0,  # reviews_per_year (simplified)
                    pred_host_listings,  # host_productivity
                    1 if pred_borough == 'Brooklyn' else 0,
                    1 if pred_borough == 'Manhattan' else 0,
                    1 if pred_borough == 'Queens' else 0,
                    1 if pred_borough == 'Staten Island' else 0,
                    1 if pred_room_type == 'Private room' else 0,
                    1 if pred_room_type == 'Shared room' else 0
                ]])
                
                # Make prediction
                predicted_price = models['random_forest'].predict(features)[0]
                
                st.success(f"💰 **Predicted Price: ${predicted_price:.0f}/night**")
                
                # Show confidence interval (approximate)
                confidence_range = 31  # Based on model MAE
                st.info(f"📊 **Confidence Range: ${predicted_price-confidence_range:.0f} - ${predicted_price+confidence_range:.0f}**")
                
                # Compare with market
                similar_listings = filtered_df[
                    (filtered_df['neighbourhood_group'] == pred_borough) &
                    (filtered_df['room_type'] == pred_room_type)
                ]['price']
                
                if len(similar_listings) > 0:
                    market_avg = similar_listings.mean()
                    percentile = (similar_listings < predicted_price).mean() * 100
                    
                    st.write(f"📈 **Market Comparison:**")
                    st.write(f"• Similar listings average: ${market_avg:.0f}")
                    st.write(f"• Your predicted price is higher than {percentile:.0f}% of similar listings")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Please ensure all fields are filled correctly.")

if __name__ == "__main__":
    main()