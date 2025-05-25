import requests
import pandas as pd

class WorldBankAPI:
    BASE_URL = "https://api.worldbank.org/v2"

    def __init__(self):
        self._country_code_map = self._load_country_code_map()

    def _load_country_code_map(self):
        url = f"{self.BASE_URL}/country"
        params = {"format": "json", "per_page": 500}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        countries = resp.json()[1]
        return {c["name"]: c["id"] for c in countries}

    def get_country_code(self, name):
        return self._country_code_map.get(name)

    def fetch_macro_indicators(self, indicator_map, country_name, start=1990, end=2023):
        code = self.get_country_code(country_name)
        if not code:
            raise ValueError(f"❌ Country not found: {country_name}")

        dfs = []
        for indicator_code, name in indicator_map.items():
            url = f"{self.BASE_URL}/country/{code}/indicator/{indicator_code}"
            params = {"format": "json", "date": f"{start}:{end}", "per_page": 1000}
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            records = resp.json()[1]
            df = pd.DataFrame([
                {"date": int(r["date"]), name: r["value"]}
                for r in records if r["value"] is not None
            ])
            df["date"] = pd.to_datetime(df["date"], format="%Y")
            df.set_index("date", inplace=True)
            dfs.append(df)

        result = pd.concat(dfs, axis=1).sort_index()
        return result
