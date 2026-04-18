import glob
import pandas as pd
import re
import datetime

def get_third_friday(year, month):
    first_day = datetime.date(year, month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day + datetime.timedelta(days=first_friday_offset)
    return first_friday + datetime.timedelta(days=14)

def audit_v2():
    files = sorted(glob.glob("data/raw/*.csv"))
    print(f"Auditing {len(files)} files...")
    
    valid_pattern = re.compile(r'^(ES|NQ|RTY)[HMUZ][0-9]$')
    
    # Store stats per instrument_id
    instr_stats = {}
    
    for i, f in enumerate(files):
        df = pd.read_csv(f, usecols=['ts_event', 'symbol', 'instrument_id'])
        df = df[df['symbol'].str.match(valid_pattern, na=False)]
        
        if df.empty:
            continue
            
        # Ensure instrument_id is int64
        df['instrument_id'] = pd.to_numeric(df['instrument_id'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['instrument_id'])
        
        agg = df.groupby(['instrument_id', 'symbol']).agg(
            first_ts=('ts_event', 'min'),
            last_ts=('ts_event', 'max'),
            count=('ts_event', 'count')
        ).reset_index()
        
        for _, row in agg.iterrows():
            iid = row['instrument_id']
            sym = row['symbol']
            
            if iid not in instr_stats:
                instr_stats[iid] = {
                    'symbol': sym,
                    'first_ts': row['first_ts'],
                    'last_ts': row['last_ts'],
                    'count': row['count']
                }
            else:
                if instr_stats[iid]['symbol'] != sym:
                    print(f"WARNING: Multiple symbols for instrument_id {iid}: {instr_stats[iid]['symbol']} and {sym}")
                instr_stats[iid]['first_ts'] = min(instr_stats[iid]['first_ts'], row['first_ts'])
                instr_stats[iid]['last_ts'] = max(instr_stats[iid]['last_ts'], row['last_ts'])
                instr_stats[iid]['count'] += row['count']

    # Process stats
    roots_count = {'ES': 0, 'NQ': 0, 'RTY': 0}
    anomalies = []
    
    month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}
    
    for iid, stats in instr_stats.items():
        sym = stats['symbol']
        match = re.match(r'^(ES|NQ|RTY)([HMUZ])([0-9])$', sym)
        root, month_code, year_digit = match.groups()
        year_digit = int(year_digit)
        
        roots_count[root] += 1
        
        first_ts = pd.to_datetime(stats['first_ts'], utc=True)
        last_ts = pd.to_datetime(stats['last_ts'], utc=True)
        
        first_year = first_ts.year
        last_year = last_ts.year
        median_year = (first_year + last_year) / 2
        
        candidate_years = [2010 + year_digit, 2020 + year_digit]
        best_year = min(candidate_years, key=lambda y: abs(y - median_year))
        
        # Rewrite symbol
        new_sym = f"{root}{month_code}{best_year}"
        stats['new_symbol'] = new_sym
        stats['root'] = root
        
        # Calculate expiry date
        expiry_date = get_third_friday(best_year, month_map[month_code])
        stats['expiry_date'] = expiry_date
        
        # Check anomaly
        expiry_ts = pd.Timestamp(expiry_date, tz='UTC')
        threshold = expiry_ts + pd.Timedelta(days=2)
        
        if last_ts > threshold and stats['count'] > 1000:
            anomalies.append({
                'instrument_id': iid,
                'symbol': new_sym,
                'first_ts': first_ts,
                'last_ts': last_ts,
                'expected_expiry': expiry_date,
                'count': stats['count']
            })

    print("\n--- Distinct instrument_id counts per root ---")
    for root, count in roots_count.items():
        print(f"{root}: {count}")
        
    print("\n--- Filtered Expiry Anomalies ---")
    if not anomalies:
        print("None found.")
    else:
        for a in sorted(anomalies, key=lambda x: x['first_ts']):
            print(f"{a['symbol']:<10} | iid: {a['instrument_id']:<8} | Last TS: {a['last_ts'].date()} > Expiry: {a['expected_expiry']} | Rows: {a['count']}")

    print("\n--- ESM0 Resolution Sample ---")
    esm0_iids = [iid for iid, s in instr_stats.items() if s['symbol'] == 'ESM0']
    for iid in esm0_iids:
        s = instr_stats[iid]
        print(f"Original: {s['symbol']} | iid: {iid:<8} | Trading: {pd.to_datetime(s['first_ts']).date()} to {pd.to_datetime(s['last_ts']).date()} -> Resolved: {s['new_symbol']}")

if __name__ == '__main__':
    audit_v2()
