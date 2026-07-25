import json
import os
from typing import Dict, Any

def finish_check() -> bool:
    """
    Check whether the last trading day has actually been processed.
        
    Returns:
        True if current_date is the last trading day, False otherwise
    """
    date_file_path = "../persistent/date.json"
    
    try:
        # Read date file
        if not os.path.exists(date_file_path):
            print(f"Warning: Date file not found: {date_file_path}")
            return False
        
        with open(date_file_path, 'r', encoding='utf-8') as f:
            date_data = json.load(f)
        
        # Get current_date and trading_days
        current_date = date_data.get('current_date')
        trading_days = date_data.get('trading_days', [])
        
        if not current_date:
            print("Warning: current_date not found in date file")
            return False
        
        if not trading_days:
            print("Warning: trading_days not found in date file")
            return False
        
        # ``current_date == last_day`` only means the final bar is ready to be
        # executed.  It must not terminate the workflow before that bar runs.
        # New sessions set ``simulation_complete`` explicitly; ``visible_through``
        # provides a safe compatibility path for sessions created before the flag.
        last_trading_day = trading_days[-1]
        if 'simulation_complete' in date_data:
            is_last = bool(date_data['simulation_complete'])
        elif date_data.get('visible_through') is not None:
            is_last = date_data.get('visible_through') == last_trading_day
        else:
            # A legacy cursor at the final date is not proof that the final bar
            # ran: the old workflow moved onto that date and then terminated
            # before executing it.  Prefer one safe final execution over silently
            # preserving that off-by-one bug.
            is_last = False
        
        if is_last:
            print(f"✅ Finish condition met: final trading day {last_trading_day} was processed")
        else:
            print(f"⏳ Current date {current_date} is not the last trading day ({last_trading_day})")
        
        return is_last
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse date file: {e}")
        return False
    except Exception as e:
        print(f"Warning: Error in finish_check: {e}")
        return False
