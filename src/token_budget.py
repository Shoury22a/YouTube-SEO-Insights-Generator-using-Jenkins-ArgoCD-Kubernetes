"""
Token Budget Tracker — Manages daily token usage for TubeRank AI.
Provides simple file-based persistence for tracking Gemini API usage.
"""

import os
import json
from datetime import datetime
from src.logger import get_logger

logger = get_logger(__name__)

BUDGET_FILE = "token_budget.json"
# Daily quota for free-tier Gemini (simulated/tracked)
DAILY_QUOTA = 1_500_000 

def _load_budget() -> dict:
    """Loads budget data from local JSON file. Resets if new day."""
    now_date = datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.exists(BUDGET_FILE):
        return {"date": now_date, "usage": 0}
    
    try:
        with open(BUDGET_FILE, "r") as f:
            data = json.load(f)
            # Daily Reset Logic
            if data.get("date") != now_date:
                logger.info(f"New day detected ({now_date}). Resetting token budget.")
                return {"date": now_date, "usage": 0}
            return data
    except Exception as e:
        logger.warning(f"Could not read budget file: {e}. Starting fresh.")
        return {"date": now_date, "usage": 0}


def _save_budget(data: dict):
    """Saves budget data to local JSON file."""
    try:
        with open(BUDGET_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save budget file: {e}")


def get_budget_status() -> dict:
    """
    Returns the current usage percentage and a user-friendly message.
    Used by the Streamlit frontend.
    """
    budget = _load_budget()
    usage = budget.get("usage", 0)
    usage_pct = min(usage / DAILY_QUOTA, 1.0)
    
    if usage_pct >= 1.0:
        msg = "🛑 Daily quota exceeded. Try again tomorrow or use a different API key."
    elif usage_pct > 0.85:
        msg = "⚠️ Near daily limit. Consider switching to linear mode."
    else:
        msg = f"You have {int((1-usage_pct)*DAILY_QUOTA):,} tokens remaining for today."
        
    return {
        "usage_pct": usage_pct,
        "message": msg,
        "current_usage": usage,
        "quota": DAILY_QUOTA
    }


def track_usage(tokens: int):
    """Increments the daily token usage count."""
    if tokens <= 0:
        return
    
    budget = _load_budget()
    budget["usage"] += tokens
    _save_budget(budget)
    logger.info(f"Budget updated: +{tokens} tokens (Total: {budget['usage']})")


def reset_budget():
    """Manually resets the budget for debugging."""
    now_date = datetime.now().strftime("%Y-%m-%d")
    _save_budget({"date": now_date, "usage": 0})
    logger.info("Token budget manually reset via UI.")
