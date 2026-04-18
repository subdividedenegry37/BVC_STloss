import glob
import pandas as pd
import re
import datetime
from collections import defaultdict

def get_expected_expiry(symbol, first_ts, last_ts):
    # Extracts the month and year from the symbol (e.g., ESU0 -> U, 0)
    match = re.match(r'^(ES|NQ|RTY)([HMUZ])([0-9])$', symbol)
    if not match:
        return None

    root, month_code, year_digit = match.groups()
    year_digit = int(year_digit)

    # Month mapping
    month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}
    target_month = month_map[month_code]

    # Infer the decade from the first_ts / last_ts
    # The CSV data spans 2010 to 2026.
    first_year = pd.to_datetime(first_ts).year
    last_year = pd.to_datetime(last_ts).year

    # We need to find the year that ends with year_digit that is closest to the trading period
    candidate_years = [2010 + year_digit, 2020 + year_digit]

    best_year = None
    for y in candidate_years:
        if y >= first_year - 1 and y <= last_year + 1:
            best_year = y
            break

    if best_year is None:
        median_year = (first_year + last_year) / 2
        best_year = min(candidate_years, key=lambda y: abs(y - median_year))

    # Calculate 3rd Friday of target_month in best_year
    first_day = datetime.date(best_year, target_month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day + datetime.timedelta(days=first_friday_offset)
    third_friday = first_friday + datetime.timedelta(days=14)

    return pd.Timestamp(third_friday, tz='UTC')

def audit_data():
    files = sorted(glob.glob("data/raw/*.csv"))
    print(f"Auditing {len(files)} files...")
    
    symbol_stats = {}
    rejected_patterns = defaultdict(int)
    
    valid_pattern = re.compile(r'^(ES|NQ|RTY)[HMUZ][0-9]$')
    
    for f in files:
        df = pd.read_csv(f, usecols=['ts_event', 'symbol'])
        
        # Track valid outrights
        valid_mask = df['symbol'].str.match(valid_pattern)
        
        df_valid = df[valid_mask]
        df_invalid = df[~valid_mask]
        
        # Aggregate valid stats
        if not df_valid.empty:
            agg = df_valid.groupby('symbol').agg(
                first_ts=('ts_event', 'min'),
                last_ts=('ts_event', 'max'),
                count=('ts_event', 'count')
            ).reset_index()
            
            for _, row in agg.iterrows():
                sym = row['symbol']
                if sym not in symbol_stats:
                    symbol_stats[sym] = {'first_ts': row['first_ts'], 'last_ts': row['last_ts'], 'count': row['count']}
                else:
                    symbol_stats[sym]['first_ts'] = min(symbol_stats[sym]['first_ts'], row['first_ts'])
                    symbol_stats[sym]['last_ts'] = max(symbol_stats[sym]['last_ts'], row['last_ts'])
                    symbol_stats[sym]['count'] += row['count']
        
        # Aggregate rejected patterns
        if not df_invalid.empty:
            # Group by symbol to see what we're rejecting
            counts = df_invalid['symbol'].value_counts()
            for sym, count in counts.items():
                rejected_patterns[sym] += count

    print("\n--- Top 20 Rejected Symbols ---")
    sorted_rejected = sorted(rejected_patterns.items(), key=lambda x: x[1], reverse=True)
    for sym, count in sorted_rejected[:20]:
        print(f"{sym:15s} : {count:>10,} rows")
        
    print("\n--- Expiry Anomalies ---")
    print(f"{'Symbol':<8} | {'First TS':<24} | {'Last TS':<24} | {'Expected Expiry':<24} | {'Flag'}")
    print("-" * 90)
    
    roots = ['ES', 'NQ', 'RTY']
    for root in roots:
        root_syms = {k: v for k, v in symbol_stats.items() if k.startswith(root)}
        sorted_syms = sorted(root_syms.items(), key=lambda x: x[1]['first_ts'])
        
        for sym, stats in sorted_syms:
            first_ts = pd.to_datetime(stats['first_ts'])
            last_ts = pd.to_datetime(stats['last_ts'])
            expected_expiry = get_expected_expiry(sym, stats['first_ts'], stats['last_ts'])

            if expected_expiry is None:
                continue

            # Flag if last_ts > expected_expiry + 7 days
            threshold = expected_expiry + pd.Timedelta(days=7)
            flag = ""
            if last_ts > threshold:
                flag = f"*** LAST TS > EXPIRY + 7D ({last_ts.date()} > {threshold.date()})"
                print(f"{sym:<8} | {str(first_ts.date()):<24} | {str(last_ts.date()):<24} | {str(expected_expiry.date()):<24} | {flag}")
            elif stats['count'] <= 10:
                flag = f"*** ONLY {stats['count']} ROWS"
                print(f"{sym:<8} | {str(first_ts.date()):<24} | {str(last_ts.date()):<24} | {str(expected_expiry.date()):<24} | {flag}")

if __name__ == '__main__':
    audit_data()
