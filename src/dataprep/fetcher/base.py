import os
import time
import requests
from dotenv import load_dotenv
import logging

load_dotenv()

class FMPClient:
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.request_count = 0

    def fetch(self, endpoint: str, params: dict = {}) -> dict:
        url = f"{self.base_url}/{endpoint}"
        headers = {"User-Agent": "Mozilla/5.0"}
        params["apikey"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        # Handle 403 Forbidden: API limits exceeded
        if response.status_code == 403:
            try:
                msg = response.json().get("Error Message", "")
                if "Limit Reach" in msg:
                    raise PermissionError(
                        "API limit reached. Please upgrade your plan or reduce request frequency."
                    )
                if "Exclusive Endpoint" in msg:
                    raise PermissionError(
                        f"Access to '{endpoint}' is restricted. Consider upgrading your plan."
                    )
            except Exception as e:
                raise PermissionError(f"Failed to access '{endpoint}': {str(e)}")

        # Handle rate limiting (429): wait for 60 seconds and retry
        while response.status_code == 429:
            print("Rate limited. Sleeping for 60 seconds...")
            time.sleep(60)
            response = requests.get(url, params=params, headers=headers)

        # Handle any other non-2xx status codes
        response.raise_for_status()
        self.request_count += 1
        logging.info(f"Successfully fetched {endpoint} from FMP API.")
        return response.json()
