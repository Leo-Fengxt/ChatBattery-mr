import os
from .domain_agent import Domain_Agent

# Prefer env vars so users don't hardcode secrets in the repo.
# mp-api itself commonly uses MP_API_KEY, and some users use MATERIALS_PROJECT_API_KEY.
MP_api_key = os.getenv("MP_API_KEY") or os.getenv("MATERIALS_PROJECT_API_KEY") or "xxx"



class Search_Agent:
    @staticmethod
    def ICSD_search(formula, ICSD_DB):
        for ICSD_formula in ICSD_DB:
            if Domain_Agent.range_match(formula, ICSD_formula):
                return True
        return False

    @staticmethod
    def MP_search(formula):
        from mp_api.client import MPRester
        try:
            with MPRester(MP_api_key) as mpr:
                # exact match
                docs = mpr.summary.search(formula=formula)
            return len(docs) >= 1
        except:
            return False
