import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set page config
st.set_page_config(page_title="Electricity Price Predictor", layout="wide")

st.title("⚡ UK Electricity Price Prediction App")
st.markdown("""
This application predicts future electricity prices based on historical weather patterns and electricity price trends.
Select a location to simulate how local weather conditions (if they were representative of the nation) might impact electricity demand and price.
""")

# --- Data Loading & Caching ---
@st.cache_data
def load_and_prep_data():
    # 1. Load Electricity Data
    elec_path = 'Electricity_values_cleaned (3).csv'
    try:
        df_elec = pd.read_csv(elec_path)
        # Select relevant columns
        df_elec = df_elec[['Year', 'Electricity_Price_p_per_kWh']].copy()
        df_elec.columns = ['Year', 'Price_pence_kWh']
        df_elec['Year'] = pd.to_numeric(df_elec['Year'], errors='coerce')
        df_elec['Price_pence_kWh'] = pd.to_numeric(df_elec['Price_pence_kWh'], errors='coerce')
        df_elec = df_elec.dropna().sort_values('Year')
    except Exception as e:
        st.error(f"Error loading electricity data: {e}")
        return None, None, None

    # 2. Load Weather Data
    weather_files = {
        'Aberporth': 'cleaned_data/Data/cleaned_data/Aberporth_cleaned.csv',
        'Armagh': 'cleaned_data/Data/cleaned_data/Armagh_cleaned.csv',
        'Chivenor': 'cleaned_data/Data/cleaned_data/Chivenor_cleaned.csv',
        'Manston': 'cleaned_data/Data/cleaned_data/Manston_cleaned.csv',
        'Wick Airport': 'cleaned_data/Data/cleaned_data/Wick-Airport_cleaned.csv'
    }
    
    weather_data = {}
    all_station_dfs = []

    for station, file_path in weather_files.items():
        try:
            df = pd.read_csv(file_path)
            # Clean cols
            cols_to_clean = ['tmax', 'tmin', 'rain_mm']
            for col in cols_to_clean:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Aggregate by Year
            agg_rules = {
                'tmax': 'mean', 'tmin': 'mean',
                'rain_mm': 'sum'
            }
            actual_agg = {k: v for k, v in agg_rules.items() if k in df.columns}
            df_yearly = df.groupby('year').agg(actual_agg).reset_index()
            df_yearly = df_yearly.rename(columns={'year': 'Year'})
            
            weather_data[station] = df_yearly
            all_station_dfs.append(df_yearly)
        except Exception as e:
            st.warning(f"Could not load data for {station}: {e}")

    # Create National Average for Training
    if all_station_dfs:
        all_weather = pd.concat(all_station_dfs)
        df_national_weather = all_weather.groupby('Year').mean().reset_index()
    else:
        st.error("No weather data loaded.")
        return None, None, None

    # Merge for Training Data
    df_train = pd.merge(df_elec, df_national_weather, on='Year', how='inner')
    
    return df_train, weather_data, df_elec

# Load Data
df_train, weather_data, df_elec_history = load_and_prep_data()

if df_train is not None:
    # --- Model Training ---
    # We train on the National Average Weather + Year -> Price
    features = ['Year', 'tmax', 'tmin', 'rain_mm']
    target = 'Price_pence_kWh'
    
    # Fill missing if any
    X = df_train[features].fillna(df_train[features].mean())
    y = df_train[target]
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X, y)

    # --- Sidebar Controls ---
    st.sidebar.header("Prediction Parameters")
    
    location = st.sidebar.selectbox("Select Location", list(weather_data.keys()))
    target_year = st.sidebar.slider("Select Year to Predict", 
                                    min_value=int(df_train['Year'].max()) + 1, 
                                    max_value=2030, 
                                    value=2025)

    # --- Prediction Logic ---
    # To predict for a future year at a specific location, we need:
    # 1. The Year (User input)
    # 2. The Weather stats. Since it's the future, we don't know the weather.
    #    Assumption: Use the average weather of the last 10 years for that SPECIFIC station.
    
    station_df = weather_data[location]
    recent_weather = station_df[station_df['Year'] >= (station_df['Year'].max() - 10)]
    
    avg_weather = recent_weather[['tmax', 'tmin', 'rain_mm']].mean()
    
    # Create input row
    input_data = pd.DataFrame({
        'Year': [target_year],
        'tmax': [avg_weather['tmax']],
        'tmin': [avg_weather['tmin']],
        'rain_mm': [avg_weather['rain_mm']],
        
    })

    # Predict
    predicted_price = model.predict(input_data)[0]

    # --- Display Results ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Forecast for {target_year} based on {location} Weather Profile")
        st.metric(label="Predicted Electricity Price", value=f"{predicted_price:.2f} p/kWh")
        
        # Visualization
        st.markdown("### Historical vs Predicted Price")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot History
        ax.plot(df_elec_history['Year'], df_elec_history['Price_pence_kWh'], label='Historical Prices', color='blue')
        
        # Plot Prediction
        ax.scatter([target_year], [predicted_price], color='red', s=100, label='Prediction', zorder=5)
        
        # Draw line to prediction
        last_year = df_elec_history.iloc[-1]['Year']
        last_price = df_elec_history.iloc[-1]['Price_pence_kWh']
        ax.plot([last_year, target_year], [last_price, predicted_price], color='red', linestyle='--')
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (p/kWh)")
        ax.set_title("Electricity Price Trend")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Weather Assumptions")
        st.info(f"Using 10-year average weather data from **{location}** to simulate conditions.")
        
        weather_metrics = {
            "Avg Max Temp": f"{avg_weather['tmax']:.1f} °C",
            "Avg Min Temp": f"{avg_weather['tmin']:.1f} °C",
            "Annual Rainfall": f"{avg_weather['rain_mm']:.0f} mm",
        }
        
        for metric, value in weather_metrics.items():
            st.write(f"**{metric}:** {value}")

    st.markdown("---")
    st.markdown("*Note: This model uses a Random Forest Regressor trained on UK national electricity prices (1970-2024) and aggregated weather data. The 'Location' feature simulates how the national price might react if the national weather resembled the climate of the selected location.*")

else:
    st.error("Failed to initialize app due to missing data.")
