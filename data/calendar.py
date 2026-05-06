"""
NSE Trading Calendar - Defines trading dates and market hours.
"""
import pandas as pd
from datetime import datetime, timedelta
import pytz

# NSE holidays (major ones, simplified - extend as needed)
NSE_HOLIDAYS = [
    # 2024
    '2024-01-26',  # Republic Day
    '2024-03-08',  # Maha Shivaratri
    '2024-03-25',  # Holi
    '2024-03-29',  # Good Friday
    '2024-04-11',  # Eid ul-Fitr
    '2024-04-17',  # Ram Navami
    '2024-04-21',  # Mahavir Jayanti
    '2024-05-23',  # Buddha Purnima
    '2024-06-17',  # Eid ul-Adha
    '2024-07-17',  # Muharram
    '2024-08-15',  # Independence Day
    '2024-08-26',  # Janmashtami
    '2024-09-16',  # Milad un-Nabi
    '2024-10-02',  # Gandhi Jayanti
    '2024-10-12',  # Dussehra
    '2024-10-31',  # Diwali
    '2024-11-01',  # Diwali (Day 2)
    '2024-11-15',  # Guru Nanak Jayanti
    '2024-12-25',  # Christmas
]

# Market hours (IST timezone)
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
IST = pytz.timezone('Asia/Kolkata')


def get_nse_holidays(start_date: str, end_date: str) -> list:
    """Return list of NSE holidays between dates."""
    holidays = []
    for holiday in NSE_HOLIDAYS:
        if start_date <= holiday <= end_date:
            holidays.append(pd.Timestamp(holiday))
    return holidays


def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get all NSE trading dates (business days excluding holidays).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DatetimeIndex of trading dates
    """
    # Get business days
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

    # Remove holidays
    holidays = get_nse_holidays(start_date, end_date)
    trading_dates = all_dates.difference(holidays)

    return trading_dates


def is_trading_day(date: pd.Timestamp) -> bool:
    """Check if a date is a trading day."""
    if date.weekday() >= 5:  # Saturday, Sunday
        return False

    date_str = date.strftime('%Y-%m-%d')
    if date_str in NSE_HOLIDAYS:
        return False

    return True


def get_market_hours(date: pd.Timestamp) -> tuple:
    """
    Get market open/close times for a date.

    Returns:
        (open_time, close_time) as timezone-aware timestamps
    """
    if not is_trading_day(date):
        return None, None

    open_time = IST.localize(datetime.combine(date, datetime.strptime(MARKET_OPEN, "%H:%M").time()))
    close_time = IST.localize(datetime.combine(date, datetime.strptime(MARKET_CLOSE, "%H:%M").time()))

    return open_time, close_time


def align_to_trading_calendar(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Align dates to nearest trading date (forward fill)."""
    trading_dates = get_trading_dates(dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d'))
    return dates.map(lambda d: trading_dates[trading_dates.searchsorted(d)])
