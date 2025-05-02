#!/usr/bin/env python
# coding: utf-8

# In[1]:


from crewai.tools import tool
import requests
from datetime import datetime, timedelta

API_KEY = "1e930e18e69e9479230a789fc81b4e87"

def get_weather_forecast_for_dates(city, available_dates, api_key):
    """Fetches weather forecast for the next available dates in a given city."""
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()

    if response.get("cod") != "200":
        return f" Error: {response.get('message')}"

    forecast_data = {}
    for entry in response["list"]:
        date = entry["dt_txt"].split(" ")[0]
        if date in available_dates:
            temp = entry["main"]["temp"]
            condition = entry["weather"][0]["description"]
            forecast_data[date] = f" Temp: {temp}°C,  Condition: {condition}"

    return forecast_data

@tool("Get Weather Forecast")
def get_weather_data(city: str):
    """Fetches weather data for the next 4 days in a given city."""
    # Get today's date and next 4 days
    today = datetime.today()
    available_dates = {today.strftime('%Y-%m-%d')}

    for i in range(1, 5):  # Add next 4 days
        next_day = today + timedelta(days=i)
        available_dates.add(next_day.strftime('%Y-%m-%d'))

    # Get weather forecast
    forecast = get_weather_forecast_for_dates(city, available_dates, API_KEY)

    if forecast:
        return '\n'.join([f" {date}: {data}" for date, data in forecast.items()])
    return " No weather data available."


# In[ ]:




