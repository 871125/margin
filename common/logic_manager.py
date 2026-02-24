
import numpy as np
import pandas as pd

def detect_swing_point(df, fail_limit):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔹 Swing Point Detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Rolling Window를 사용하여 국소적 고점(High)과 저점(Low)을 탐색
    # 2. 한 캔들에서 고점/저점이 동시에 발생할 경우 캔들 색상(양봉/음봉)에 따라 순서 결정
    # 3. High -> Low -> High 순서가 유지되도록 필터링 (Alternation)

    df = df.copy()
    df = df.sort_index(ascending=True)
    df["swing_high"] = np.nan
    df["swing_low"]  = np.nan

    window = fail_limit * 2 + 1

    df['max_rolling'] = df['high'].rolling(window=window, center=True, min_periods=1).max()
    df['min_rolling'] = df['low'].rolling(window=window, center=True, min_periods=1).min()

    candidates = []
    for i in range(len(df)):
        idx = df.index[i]
        
        is_high = df['high'].iloc[i] == df['max_rolling'].iloc[i]
        is_low  = df['low'].iloc[i] == df['min_rolling'].iloc[i]

        if is_high and is_low:
            # 하나의 캔들에서 High/Low 동시 발생 시 캔들 색상으로 순서 결정
            if df['close'].iloc[i] > df['open'].iloc[i]:
                # 양봉: 저점 찍고 고점으로 -> Low, High 순서
                candidates.append({'idx': idx, 'type': 'Low', 'val': df['low'].iloc[i]})
                candidates.append({'idx': idx, 'type': 'High', 'val': df['high'].iloc[i]})
            else:
                # 음봉: 고점 찍고 저점으로 -> High, Low 순서
                candidates.append({'idx': idx, 'type': 'High', 'val': df['high'].iloc[i]})
                candidates.append({'idx': idx, 'type': 'Low', 'val': df['low'].iloc[i]})
        elif is_high:
            candidates.append({'idx': idx, 'type': 'High', 'val': df['high'].iloc[i]})
        elif is_low:
            candidates.append({'idx': idx, 'type': 'Low', 'val': df['low'].iloc[i]})

    if not candidates:
        return df.sort_index(ascending=False)

    final_swings = []
    current_swing = candidates[0]

    for i in range(1, len(candidates)):
        next_cand = candidates[i]

        if next_cand['type'] == current_swing['type']:
            # If same type, keep the better one
            if current_swing['type'] == 'High':
                if next_cand['val'] > current_swing['val']:
                    current_swing = next_cand
            else:
                if next_cand['val'] < current_swing['val']:
                    current_swing = next_cand
        else:
            # If different type, confirm current and switch
            final_swings.append(current_swing)
            current_swing = next_cand
    
    final_swings.append(current_swing)

    # Assign to DataFrame
    for s in final_swings:
        if s['type'] == 'High':
            df.at[s['idx'], 'swing_high'] = s['val']
        else:
            df.at[s['idx'], 'swing_low'] = s['val']

    df.drop(columns=['max_rolling', 'min_rolling'], inplace=True)
    
    # Return in descending order to match original behavior
    return df.sort_index(ascending=False)

def calc_trend(df):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔹 Trend Calculation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 최근 2개의 Swing High/Low를 비교하여 추세 결정
    #    - 고점 상승 & 저점 상승 => Up Trend
    #    - 고점 하락 & 저점 하락 => Down Trend
    # 2. 추세가 명확하지 않으면 Range(횡보)로 설정

    df = df.copy()
    df = df.sort_index(ascending=True)
    df["status"] = None

    recent_highs = []
    recent_lows = []
    bef_status = range
    
    for i in range(len(df)):
        s_high = df['swing_high'].iloc[i]
        s_low = df['swing_low'].iloc[i] 

        if pd.isna(s_high) and pd.isna(s_low):
            continue

        if not pd.isna(s_high):
            recent_highs.append(s_high)
            if len(recent_highs) > 2:
                recent_highs.pop(0)
            
        if not pd.isna(s_low):
            recent_lows.append(s_low)
            if len(recent_lows) > 2:
                recent_lows.pop(0)
            
        if len(recent_highs) == 2 and len(recent_lows) == 2:
            is_up = recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]
            is_down = recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]
        else:
            continue

        if is_up and bef_status != 'up':
            df.at[i, 'status'] = 'up'
            bef_status = 'up'
        
        if is_down and bef_status != 'down': 
            df.at[i, 'status']= 'down'
            bef_status = 'down'
        
        if not is_down and not is_up and bef_status != 'range':
            df.at[i, 'status'] = 'range'
            bef_status = 'range'

    return df.sort_index(ascending=False)

def detect_reversal_candles(df):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔹 Candlestick Pattern Recognition
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 캔들의 Body, Wick, Range 등 기본 속성 계산
    # 2. 주요 반전 패턴 탐지: Doji, Hammer/Shooting Star, Engulfing
    # 3. 거래량(Volume) 조건을 추가하여 신뢰도 향상

    df = df.copy()
    df = df.sort_index(ascending=False)

    df["body"] = (df["close"] - df["open"]).abs()
    df["upperWick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lowerWick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]

    df["highest_6"] = df["high"].rolling(6).max().shift(1)
    df["lowest_6"] = df["low"].rolling(6).min().shift(1)

    df['volMa20'] = df["volume"].rolling(20).mean()
    movingAvgPct = 1.2

    df["bearDoji"] = ((df["body"] <= df["range"] * 0.10) & 
                      (df["upperWick"] > df["lowerWick"]) & 
                      (df["high"] >= df["highest_6"]) &
                      (df['volume'] >= df['volMa20'] * movingAvgPct ))
    df["bullDoji"] = ((df["body"] <= df["range"] * 0.10) & 
                      (df["lowerWick"] > df["upperWick"]) &
                      (df["low"] <= df["lowest_6"])&
                      (df['volume'] >= df['volMa20'] * movingAvgPct))

    df['bullHS'] = ((df[['high', 'low']].mean(axis =1) <= df[['close', 'open']].min(axis=1))& 
                    (df['lowerWick'] >= df['body'] *2) & (df['upperWick'] <= df['body'] * 0.25) & 
                    (df["low"] <= df["lowest_6"])&
                    (df['volume'] >= df['volMa20'] * movingAvgPct))
    df['bearHS'] = ((df[['high', 'low']].mean(axis =1) >= df[['close', 'open']].min(axis=1)) & (df['upperWick'] >= df['body'] *2) & 
                    (df['lowerWick'] <= df['body'] * 0.25) & 
                    (df["high"] >= df["highest_6"]) &
                    (df['volume'] >= df['volMa20'] * movingAvgPct))

    lowest_5_shift2  = df["low"].rolling(5).min().shift(2)
    highest_5_shift2 = df["high"].rolling(5).max().shift(2)

    df["bullEngulf"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] >= df["open"].shift(1)) &
        (df["open"] <= df["close"].shift(1)) &
        (
            (df["low"].shift(1) <= df["lowest_6"]) |
            (df["low"] <= lowest_5_shift2)
        ) &
        (df['volume'] >= df['volMa20'] * movingAvgPct)
    )

    df["bearEngulf"] = (
        (df["close"] < df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["close"] <= df["open"].shift(1)) &
        (df["open"] >= df["close"].shift(1)) &
        (
            (df["high"].shift(1) >= df["highest_6"]) |
            (df["high"] >= highest_5_shift2)
        ) &
        (df['volume'] >= df['volMa20'] * movingAvgPct)
    )
    return df

def find_volume_profile(df, min_candles=30, lookback=50, vol_threshold=0.1, break_threshold=0.015):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔹 Volume Profile / Consolidation Zones
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 가격 변동폭(Break)을 감지하여 횡보 구간 식별
    # 2. 일정 기간(min_candles) 이상 유지된 구간을 Zone으로 설정
    # 3. Zone 내부의 고점/저점을 기록

    df = df.copy()
    df = df.sort_index(ascending=True)

    df['lookback_max_high'] = df['high'].rolling(window=lookback, min_periods = 1).max().shift(1)
    df['lookback_min_low']  = df['low'].rolling(window=lookback, min_periods = 1).min().shift(1)

    df['break'] = ((df['high']/df['close'].shift(1)) >= (1+break_threshold)) | ((df['close'].shift(1)/df['low']) >=(1+break_threshold))

    df['zone_id'] = np.nan
    df['zone_high'] = np.nan
    df['zone_low'] = np.nan

    current_zone = None
    zone_id = 0

    for i in range(len(df)):
        idx = df.index[i]
        row = df.iloc[i]

        if row['break']:
            if current_zone:
                if len(current_zone['indices']) >= min_candles:
                    zone_id += 1
                    df.loc[current_zone['indices'], 'zone_id'] = zone_id
                    df.loc[current_zone['indices'], 'zone_high'] = current_zone['max']
                    df.loc[current_zone['indices'], 'zone_low'] = current_zone['min']
                current_zone = None
            continue

        if current_zone is None:
            current_zone = {'min': row['low'], 'max': row['high'], 'indices': [idx]}
        else:
            new_min = min(current_zone['min'], row['low'])
            new_max = max(current_zone['max'], row['high'])
            
            if (new_max - new_min) / new_min <= vol_threshold:
                current_zone['min'] = new_min
                current_zone['max'] = new_max
                current_zone['indices'].append(idx)
            else:
                if len(current_zone['indices']) >= min_candles:
                    zone_id += 1
                    df.loc[current_zone['indices'], 'zone_id'] = zone_id
                    df.loc[current_zone['indices'], 'zone_high'] = current_zone['max']
                    df.loc[current_zone['indices'], 'zone_low'] = current_zone['min']
                current_zone = {'min': row['low'], 'max': row['high'], 'indices': [idx]}

    if current_zone and len(current_zone['indices']) >= min_candles:
        zone_id += 1
        df.loc[current_zone['indices'], 'zone_id'] = zone_id
        df.loc[current_zone['indices'], 'zone_high'] = current_zone['max']
        df.loc[current_zone['indices'], 'zone_low'] = current_zone['min']

    return df.sort_index(ascending=False)

def find_zone(df):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔹 Trend Zones & Fibonacci Retracement
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 연속된 추세(Up/Down) 구간을 블록으로 그룹화
    # 2. 각 추세 구간 내의 Swing Point를 식별하여 Zone 설정
    # 3. 마지막 유효 추세 구간에 대해 피보나치 되돌림(38.2%, 50%, 61.8%) 계산

    df = df.copy()
    df = df.sort_index(ascending=True)

    temp_status = df['status'].ffill()
    
    groups = temp_status.ne(temp_status.shift()).cumsum()

    # Find the last group that is 'up' or 'down'
    group_status = temp_status.groupby(groups).first()
    valid_keys = group_status[group_status.isin(['up', 'down'])].index
    last_valid_key = valid_keys[-1] if not valid_keys.empty else None
    
    for name, group in df.groupby(groups):
        if group.empty: continue
        
        first_idx = group.index[0]
        last_idx = group.index[-1]
        status = temp_status.loc[first_idx]
        
        if status in ['up', 'down']:
            points = []
            prefix = ""
            
            if status == 'up':
                mask = (df['swing_low'].notna()) & (df.index <= last_idx)
                all_lows = df.index[mask]
                
                points_before_or_at = all_lows[all_lows <= first_idx]
                points_after = all_lows[all_lows > first_idx]
                
                points = sorted(points_before_or_at[-2:].tolist() + points_after.tolist())
                prefix = "trend_up_zone_"
                
            elif status == 'down':
                mask = (df['swing_high'].notna()) & (df.index <= last_idx)
                all_highs = df.index[mask]
                
                points_before_or_at = all_highs[all_highs <= first_idx]
                points_after = all_highs[all_highs > first_idx]
                
                points = sorted(points_before_or_at[-2:].tolist() + points_after.tolist())
                prefix = "trend_down_zone_"
            
            for i, pt_idx in enumerate(points):
                col_high = f'{prefix}{i}_high'
                col_low = f'{prefix}{i}_low'
                
                ref_high = df.loc[pt_idx, 'high']
                ref_low = df.loc[pt_idx, 'low']
                
                fill_idx = df.loc[pt_idx:last_idx].index

                df.loc[fill_idx, col_high] = ref_high
                df.loc[fill_idx, col_low] = ref_low

                if name == last_valid_key:
                    if status == 'up':
                        mask_next = (df.index > pt_idx) & (df['swing_high'].notna())
                        next_highs = df.index[mask_next]
                        
                        if not next_highs.empty:
                            calc_high = df.loc[next_highs[0], 'swing_high']
                            height = calc_high - ref_low

                            df.loc[fill_idx, f'{prefix}{i}_fib_382'] = ref_low + height * 0.382
                            df.loc[fill_idx, f'{prefix}{i}_fib_500'] = ref_low + height * 0.5
                            df.loc[fill_idx, f'{prefix}{i}_fib_618'] = ref_low + height * 0.618
                    elif status == 'down':
                        mask_next = (df.index > pt_idx) & (df['swing_low'].notna())
                        next_lows = df.index[mask_next]
                        
                        if not next_lows.empty:
                            calc_low = df.loc[next_lows[0], 'swing_low']
                            height = ref_high - calc_low

                            df.loc[fill_idx, f'{prefix}{i}_fib_382'] = ref_high - height * 0.382
                            df.loc[fill_idx, f'{prefix}{i}_fib_500'] = ref_high - height * 0.5
                            df.loc[fill_idx, f'{prefix}{i}_fib_618'] = ref_high - height * 0.618

    return df.sort_index(ascending=False)

        



def price_action(df):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔹 Price Action Analysis Pipeline
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 전체 분석 로직을 순차적으로 실행
    df = detect_swing_point(df, fail_limit=14)
    df = calc_trend(df)
    df = detect_reversal_candles(df)
    df = find_volume_profile(df)
    df = find_zone(df)

    return df 
