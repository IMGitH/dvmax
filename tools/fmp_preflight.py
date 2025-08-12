import sys
from src.dataprep.fetcher._fmp_client import fmp_get, FMPAuthError, FMPPlanError, FMPRateLimitError, FMPServerError

try:
    _ = fmp_get("/api/v3/profile/AAPL", {"limit": 1})  # cheap endpoint
    _ = fmp_get("/api/v3/ratios/AAPL", {"limit": 1})   # the one you use
    print("OK")
    sys.exit(0)
except FMPAuthError as e:
    print(f"AUTH:{e}"); sys.exit(20)
except FMPPlanError as e:
    print(f"PLAN:{e}"); sys.exit(21)
except FMPRateLimitError as e:
    print(f"RATE:{e}"); sys.exit(22)
except FMPServerError as e:
    print(f"SERVER:{e}"); sys.exit(23)
except Exception as e:
    print(f"OTHER:{e}"); sys.exit(24)
