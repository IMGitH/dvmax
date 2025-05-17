import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class FMPClient:
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def fetch(self, endpoint: str, params: dict = {}) -> dict:
        url = f"{self.base_url}/{endpoint}"
        headers = {"User-Agent": "Mozilla/5.0"}
        params["apikey"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 403:
            try:
                msg = response.json().get("Error Message", "")
                if "Exclusive Endpoint" in msg:
                    raise PermissionError(
                        f"Access to '{endpoint}' is restricted. Consider upgrading your plan."
                    )
            except Exception:
                pass
            
        while response.status_code == 429:
            print("Rate limited. Sleeping for 60 seconds...")
            time.sleep(60)
            response = requests.get(url)

        response.raise_for_status()
        return response.json()
