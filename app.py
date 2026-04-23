import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_models():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Model files not found! Please train the model first.")
        return None, None, None

model, scaler, feature_columns = load_models()

# Title and description
st.title("🏠 House Price Prediction System")
st.markdown("### Predict the market value of your house with AI")
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=80)
    st.title("About")
    st.info(
        """
        This AI-powered tool predicts house prices based on various features
        like area, number of rooms, amenities, and location preferences.
        
        **Model Performance:**
        - Random Forest Regressor
        - R² Score: 0.85 (85% accuracy)
        
        **Features used:**
        - Property characteristics
        - Amenities
        - Location advantages
        """
    )
    
    st.markdown("---")
    st.markdown("### How to use?")
    st.markdown("""
    1. Fill in the property details
    2. Click 'Predict Price'
    3. Get instant market value estimation
    """)

# Create two columns for input layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏡 Property Details")
    
    # Basic property features
    area = st.number_input(
        "Area (sq ft)",
        min_value=500,
        max_value=20000,
        value=2000,
        step=100,
        help="Total area of the house in square feet"
    )
    
    bedrooms = st.selectbox(
        "Number of Bedrooms",
        options=[1, 2, 3, 4, 5, 6],
        index=2,
        help="Total number of bedrooms"
    )
    
    bathrooms = st.selectbox(
        "Number of Bathrooms",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Total number of bathrooms"
    )
    
    stories = st.selectbox(
        "Number of Stories",
        options=[1, 2, 3, 4],
        index=1,
        help="Number of floors in the house"
    )
    
    parking = st.selectbox(
        "Parking Spaces",
        options=[0, 1, 2, 3, 4],
        index=1,
        help="Number of parking spaces available"
    )

with col2:
    st.markdown("### 🛋️ Amenities & Features")
    
    # Binary features
    mainroad = st.selectbox(
        "Connected to Main Road",
        options=['yes', 'no'],
        index=0,
        help="Is the property connected to main road?"
    )
    
    guestroom = st.selectbox(
        "Has Guest Room",
        options=['yes', 'no'],
        index=1,
        help="Does the house have a guest room?"
    )
    
    basement = st.selectbox(
        "Has Basement",
        options=['yes', 'no'],
        index=1,
        help="Does the house have a basement?"
    )
    
    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        options=['yes', 'no'],
        index=1,
        help="Does the house have hot water heating system?"
    )
    
    airconditioning = st.selectbox(
        "Air Conditioning",
        options=['yes', 'no'],
        index=0,
        help="Does the house have air conditioning?"
    )
    
    prefarea = st.selectbox(
        "Preferred Area",
        options=['yes', 'no'],
        index=0,
        help="Is the house in a preferred/residential area?"
    )
    
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        options=['unfurnished', 'semi-furnished', 'furnished'],
        index=2,
        help="What is the furnishing status?"
    )

# Calculate derived features automatically
total_rooms = bedrooms + bathrooms
room_ratio = bedrooms / max(bathrooms, 1)
luxury_score = sum([
    1 if airconditioning == 'yes' else 0,
    1 if hotwaterheating == 'yes' else 0,
    1 if guestroom == 'yes' else 0,
    1 if prefarea == 'yes' else 0
])
parking_premium = parking * (1 if prefarea == 'yes' else 0)
area_per_bedroom = area / max(bedrooms, 1)

# Display metrics in a row
st.markdown("---")
st.markdown("### 📊 Property Metrics")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Total Rooms", f"{total_rooms}", help="Bedrooms + Bathrooms")

with metric_col2:
    st.metric("Room Ratio", f"{room_ratio:.2f}", help="Bedrooms to Bathrooms ratio")

with metric_col3:
    st.metric("Luxury Score", f"{luxury_score}/4", help="Premium features count")

with metric_col4:
    st.metric("Area per Bedroom", f"{area_per_bedroom:.0f} sq ft", help="Average area per bedroom")

with metric_col5:
    st.metric("Parking Premium", "Yes" if parking_premium > 0 else "No", help="Parking in preferred area")

# Prediction button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 PREDICT HOUSE PRICE", use_container_width=True)

# Make prediction
if predict_button:
    if model is not None and scaler is not None:
        # Prepare input data
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': 1 if mainroad == 'yes' else 0,
            'guestroom': 1 if guestroom == 'yes' else 0,
            'basement': 1 if basement == 'yes' else 0,
            'hotwaterheating': 1 if hotwaterheating == 'yes' else 0,
            'airconditioning': 1 if airconditioning == 'yes' else 0,
            'parking': parking,
            'prefarea': 1 if prefarea == 'yes' else 0,
            'furnishingstatus': 0 if furnishingstatus == 'unfurnished' else (1 if furnishingstatus == 'semi-furnished' else 2),
            'total_rooms': total_rooms,
            'room_ratio': room_ratio,
            'luxury_score': luxury_score,
            'parking_premium': parking_premium,
            'area_per_bedroom': area_per_bedroom
        }
        
        # Create DataFrame with the same columns as training
        input_df = pd.DataFrame([input_data])
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        
        # Create three columns for displaying prediction
        pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
        
        with pred_col2:
            # Convert to different units
            price_crore = prediction / 10000000
            price_lakh = prediction / 100000
            price_usd = prediction / 83  # Approximate conversion
            
            st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color: #FF4B4B; margin-bottom: 20px;'>💰 Estimated Price</h2>
                <h1 style='color: #FF4B4B; font-size: 48px; margin-bottom: 10px;'>₹{prediction:,.0f}</h1>
                <p style='font-size: 18px; color: #666;'>
                    (₹{price_crore:.2f} Crore | ₹{price_lakh:.2f} Lakh)
                </p>
                <p style='font-size: 14px; color: #888;'>
                    ≈ ${price_usd:,.0f} USD
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add price range indicator
        st.markdown("### 📈 Price Analysis")
        
        # Calculate price range (±15%)
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction,
            title = {'text': "Price Range (in ₹)"},
            delta = {'reference': prediction},
            gauge = {
                'axis': {'range': [None, upper_bound]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [lower_bound, prediction], 'color': "lightgray"},
                    {'range': [prediction, upper_bound], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature contributions (if using tree-based model)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🔍 Key Factors Influencing This Price")
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Get top 5 features
            top_features = importance_df.head(5)
            
            cols = st.columns(5)
            for idx, (_, row) in enumerate(top_features.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>{row['Feature'].replace('_', ' ').title()}</h4>
                        <h3>{row['Importance']*100:.1f}%</h3>
                        <small>importance</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Success message
        st.success("✅ Prediction completed successfully!")
        
        # Add download button for report
        report_data = {
            "Predicted Price (₹)": prediction,
            "Predicted Price (Crore)": price_crore,
            "Predicted Price (Lakh)": price_lakh,
            "Area (sq ft)": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Stories": stories,
            "Parking": parking,
            "Main Road": mainroad,
            "Guest Room": guestroom,
            "Basement": basement,
            "Air Conditioning": airconditioning,
            "Preferred Area": prefarea,
            "Furnishing": furnishingstatus
        }
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv,
            file_name="house_price_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.error("❌ Model not loaded properly. Please check the model files.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Powered by Machine Learning | Random Forest Regressor</p>
    <p>© 2024 House Price Prediction System</p>
</div>
""", unsafe_allow_html=True)