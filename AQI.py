import requests
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st


API_KEY = "a81aba2cb56473a54faf866494895024"

def get_coordinates(city):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    geo_response = requests.get(geo_url).json()
    lat, lon = geo_response[0]['lat'], geo_response[0]['lon']
    return lat, lon


def fetch_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url).json()
    records = []
    for item in response['list']:
        dt = datetime.datetime.fromtimestamp(item['dt'])
        aqi = item['main']['aqi']
        pollutants = item['components']
        pollutants['AQI'] = aqi
        pollutants['datetime'] = dt
        records.append(pollutants)
    return pd.DataFrame(records)

def run_regression(df):
    X = df[['pm2_5','pm10','co','no2','o3','so2']]
    y = df['AQI']
    model = LinearRegression()
    model.fit(X, y)
    return model, model.score(X, y)

st.title("🌍 AQI Agent Dashboard")

city = st.text_input("Enter a city name (e.g., Hyderabad,IN):")

if city:
    lat, lon = get_coordinates(city)
    df = fetch_aqi(lat, lon)
    model, r2 = run_regression(df)

    st.write(f"**City:** {city} (lat={lat}, lon={lon})")
    st.write(f"**Regression R² Score:** {r2:.2f}")
    st.write(f"**Max AQI:** {df['AQI'].max()}")

    if df['AQI'].max() >= 4:
        st.error("⚠️ Poor air quality expected. Limit outdoor activity.")
    else:
        st.success("✅ Air quality is acceptable.")

    # Plot AQI forecast
    st.line_chart(df.set_index("datetime")["AQI"])
