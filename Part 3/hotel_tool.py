#!/usr/bin/env python
# coding: utf-8

# In[17]:


from crewai.tools import tool
import requests
import time

API_KEY = "SvPcGNyRpiZGdf3b0QZjtqIwlwNeE1wy"
API_SECRET = "AoShu7xewCsKuwsl"

def get_access_token(api_key, api_secret):
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": api_secret
    }
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()['access_token']

@tool("Search Hotels")
def get_hotel_data_combined(city_code: str, check_in: str, check_out: str, adults: int):
    """Searches for hotels in a given city and date range, returning offers and ratings."""
    token = get_access_token(API_KEY, API_SECRET)

    # Get hotel IDs
    hotel_id_url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
    hotel_id_res = requests.get(hotel_id_url, headers={"Authorization": f"Bearer {token}"}, params={"cityCode": city_code})
    hotel_ids = [h['hotelId'] for h in hotel_id_res.json().get('data', [])][:10]

    if not hotel_ids:
        return "No hotel data found."

    # Get hotel offers
    offer_url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
    offer_params = {
        "hotelIds": ','.join(hotel_ids),
        "checkInDate": check_in,
        "checkOutDate": check_out,
        "adults": adults
    }
    offer_res = requests.get(offer_url, headers={"Authorization": f"Bearer {token}"}, params=offer_params)
    offers = offer_res.json().get("data", [])

    # Get ratings
    rating_url = "https://test.api.amadeus.com/v2/e-reputation/hotel-sentiments"
    ratings = {}
    for i in range(0, len(hotel_ids), 3):
        chunk = hotel_ids[i:i+3]
        rating_res = requests.get(rating_url, headers={"Authorization": f"Bearer {token}"}, params={"hotelIds": ','.join(chunk)})
        if rating_res.status_code == 200:
            for item in rating_res.json().get("data", []):
                ratings[item['hotelId']] = {
                    "rating": item.get("overallRating"),
                    "reviews": item.get("numberOfReviews")
                }
        time.sleep(0.2)

    # Merge + Format result
    output = ""
    for hotel_offer in offers:
        hotel = hotel_offer["hotel"]
        hotel_id = hotel["hotelId"]
        output += f"\n {hotel.get('name')} (ID: {hotel_id})\n"
        output += f" {hotel.get('cityCode')} |  {hotel.get('latitude')}, {hotel.get('longitude')}\n"
        if ratings.get(hotel_id):
            output += f" Rating: {ratings[hotel_id]['rating']} ({ratings[hotel_id]['reviews']} reviews)\n"
        else:
            output += " Rating: Not available\n"

        for offer in hotel_offer.get("offers", []):
            output += f"   {offer['checkInDate']} → {offer['checkOutDate']}\n"
            output += f"   {offer['price']['total']} {offer['price']['currency']}\n"
            room = offer.get("room", {})
            desc = room.get("description", {}).get("text", "")
            output += f"   Room: {room.get('type', 'N/A')} - {desc}\n"
            output += "  -----------------------------\n"

    return output or "No hotel offers found."


# In[ ]:




