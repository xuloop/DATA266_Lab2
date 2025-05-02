#!/usr/bin/env python
# coding: utf-8

# In[3]:


from crewai.tools import tool
from amadeus_api import get_access_token
import requests

@tool("Search Flights")
def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None,
                   adults: int = 2, max_results: int = 5, non_stop: bool = False) -> str:
    """Search flights using Amadeus API with optional nonstop filter"""
    print(f"Tool called with: origin={origin}, destination={destination}, departure_date={departure_date}, return_date={return_date}, adults={adults}, max_results={max_results}, non_stop={non_stop}")

    token = get_access_token()
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "adults": adults,
        "max": max_results,
        "nonStop": str(non_stop).lower()  # Amadeus expects 'true' or 'false' as strings
    }
    if return_date:
        params["returnDate"] = return_date

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception("Flight search failed:", response.text)

    flights = response.json().get("data", [])
    if not flights:
        return "No flights found."

    formatted_results = []

    for i, f in enumerate(flights[:max_results]):
        try:
            price = f['price']['total']
            currency = f['price']['currency']
            itineraries = f['itineraries']

            flight_info = [f"Flight Option {i + 1}:", f"Total Price: {price} {currency}"]

            for leg_idx, leg in enumerate(itineraries):
                leg_type = "Departure" if leg_idx == 0 else "Return"
                flight_info.append(f"\nLeg {leg_idx + 1} ({leg_type})")
                flight_info.append(f"Duration: {leg.get('duration', 'N/A')}")

                for j, segment in enumerate(leg['segments']):
                    carrier = segment['carrierCode']
                    flight_number = segment['number']
                    departure = segment['departure']
                    arrival = segment['arrival']

                    flight_info.append(f"\nSegment {j + 1}:")
                    flight_info.append(f"From: {departure['iataCode']} at {departure['at']}")
                    flight_info.append(f"To: {arrival['iataCode']} at {arrival['at']}")
                    flight_info.append(f"Carrier: {carrier}, Flight: {flight_number}")

            formatted_results.append("\n".join(flight_info))

        except Exception as e:
            print(f"Skipping flight due to error: {e}")
            continue

    return "\n\n---\n\n".join(formatted_results) or "No valid flights found."


# In[ ]:




