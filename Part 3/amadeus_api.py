#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
from dotenv import load_dotenv

CLIENT_ID = "SvPcGNyRpiZGdf3b0QZjtqIwlwNeE1wy"
CLIENT_SECRET = "AoShu7xewCsKuwsl"

def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise Exception("Failed to get access token:", response.text)
    return response.json()["access_token"]


# In[ ]:




