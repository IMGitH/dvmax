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
        """
        Fetches multiple macroeconomic indicators for a given country and year range.

        Args:
            indicator_map (dict): {"indicator_code": "human_name"}
            country_name (str): e.g., "United States"
            start (int): Start year
            end (int): End year

        Returns:
            pd.DataFrame with columns = human_name, index = datetime(year)
        """
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

if __name__ == "__main__":
    macro = WorldBankAPI()

    indicators = {
        "NY.GDP.MKTP.CD": "GDP (USD)",  # Measures total economic output; a shrinking GDP can signal lower corporate earnings and higher cut risk.

        "FP.CPI.TOTL.ZG": "Inflation (%)",  # High inflation erodes real cash flow and investor yield, increasing pressure on dividends.

        "SL.UEM.TOTL.ZS": "Unemployment (%)",  # Proxy for recession risk; high unemployment correlates with weak consumption and potential payout suspensions.

        "NE.EXP.GNFS.ZS": "Exports (% GDP)",  # Indicates external demand strength; export-heavy economies may better support dividend-paying firms in downturns.

        "NE.CON.PRVT.ZS": "Private Consumption (% GDP)"  # Reflects domestic consumer demand; weak consumption signals earnings pressure in consumer-driven sectors.
    }

    df_macro = macro.fetch_macro_indicators(indicators, "United States", start=2000, end=2023)

    print(df_macro.tail())