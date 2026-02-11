# swing point
# moving average
# candle pattern




# chart에서 매물대 찾기(supply, demand)
## Supply: 강한 하락 이전 가격대
## Demand : 강한 상승 이전 가격대
# candle pattern 만들기
## 잉걸핑 : https://cafe.naver.com/f-e/cafes/31364126/articles/520?boardtype=L&menuid=2&referrerAllArticles=false&page=2
## Hammer : https://cafe.naver.com/f-e/cafes/31364126/articles/521?boardtype=L&menuid=2&referrerAllArticles=false&page=2
## Doji : https://cafe.naver.com/f-e/cafes/31364126/articles/522?boardtype=L&menuid=2&referrerAllArticles=false&page=2
## Morning Star/ Evenig Star : https://cafe.naver.com/f-e/cafes/31364126/articles/523?boardtype=L&menuid=2&referrerAllArticles=false&page=2
## 적삼병/흑삼병 : https://cafe.naver.com/f-e/cafes/31364126/articles/524?boardtype=L&menuid=2&referrerAllArticles=false&page=2
# Moving Avg
# MACD ? 
# RSI/ATR
# 
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

def detect_swing_point(df, fail_limit):
    df = df.copy()
    # Calculate in chronological order
    df = df.sort_index(ascending=True)
    df["swing_high"] = np.nan
    df["swing_low"]  = np.nan

    # Window size for local extrema (left + right + center)
    window = fail_limit * 2 + 1

    # Find local peaks and valleys
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

    # Filter for alternation
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
    df = df.copy()
    # Calculate in chronological order (Past -> Future)
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
    df = df.copy()
    df = df.sort_index(ascending=False)

    #━━━━━━━━━━━━━━━━━━━
    # 캔들 기본 계산
    #━━━━━━━━━━━━━━━━━━━
    df["body"] = (df["close"] - df["open"]).abs()
    df["upperWick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lowerWick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]

    df["highest_6"] = df["high"].rolling(6).max().shift(1)
    df["lowest_6"] = df["low"].rolling(6).min().shift(1)

    #━━━━━━━━━━━━━━━━━━━
    # Volume Moving Average
    #━━━━━━━━━━━━━━━━━━━
    df['volMa20'] = df["volume"].rolling(20).mean()
    movingAvgPct = 1.2

    #━━━━━━━━━━━━━━━━━━━
    # Doji
    #━━━━━━━━━━━━━━━━━━━
    df["bearDoji"] = ((df["body"] <= df["range"] * 0.10) & 
                      (df["upperWick"] > df["lowerWick"]) & 
                      (df["high"] >= df["highest_6"]) &
                      (df['volume'] >= df['volMa20'] * movingAvgPct ))
    df["bullDoji"] = ((df["body"] <= df["range"] * 0.10) & 
                      (df["lowerWick"] > df["upperWick"]) &
                      (df["low"] <= df["lowest_6"])&
                      (df['volume'] >= df['volMa20'] * movingAvgPct))

    #━━━━━━━━━━━━━━━━━━━
    # Hammer / Shooting star     
    #━━━━━━━━━━━━━━━━━━━
    df['bullHS'] = ((df[['high', 'low']].mean(axis =1) <= df[['close', 'open']].min(axis=1))& 
                    (df['lowerWick'] >= df['body'] *2) & (df['upperWick'] <= df['body'] * 0.25) & 
                    (df["low"] <= df["lowest_6"])&
                    (df['volume'] >= df['volMa20'] * movingAvgPct))
    df['bearHS'] = ((df[['high', 'low']].mean(axis =1) >= df[['close', 'open']].min(axis=1)) & (df['upperWick'] >= df['body'] *2) & 
                    (df['lowerWick'] <= df['body'] * 0.25) & 
                    (df["high"] >= df["highest_6"]) &
                    (df['volume'] >= df['volMa20'] * movingAvgPct))

    #━━━━━━━━━━━━━━━━━━━
    # Engulfing
    #━━━━━━━━━━━━━━━━━━━
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
    df = df.copy()
    # Calculate in chronological order (Past -> Future)
    df = df.sort_index(ascending=True)

    # 현재 row를 제외한 이전 lookback 기간의 High Max, Low Min
    # shift(1)을 사용하여 현재 캔들을 제외하고 이전 데이터만 포함
    df['lookback_max_high'] = df['high'].rolling(window=lookback, min_periods = 1).max().shift(1)
    df['lookback_min_low']  = df['low'].rolling(window=lookback, min_periods = 1).min().shift(1)

    # 변동폭이 앞의 lookback개 캔들의 변동폭의 break_threshold 이상일 경우 filtering
    # df['break'] = (df['high']-df['low'])/(df['lookback_max_high']-df['lookback_min_low']) >= break_threshold
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



def price_action(df):
    df = detect_swing_point(df, fail_limit=5)
    df = calc_trend(df)
    df = detect_reversal_candles(df)
    df = find_volume_profile(df)

    return df 
