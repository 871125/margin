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
    df["status"] = "range"

    recent_highs = []
    recent_lows = []
    
    current_status = "range"

    for i in range(len(df)):
        idx = df.index[i]
        s_high = df['swing_high'].iloc[i]
        s_low = df['swing_low'].iloc[i]
        
        update_trend = False

        if not pd.isna(s_high):
            recent_highs.append((idx, s_high))
            if len(recent_highs) > 2:
                recent_highs.pop(0)
            update_trend = True
            
        if not pd.isna(s_low):
            recent_lows.append((idx, s_low))
            if len(recent_lows) > 2:
                recent_lows.pop(0)
            update_trend = True
            
        if update_trend and len(recent_highs) == 2 and len(recent_lows) == 2:
            h1_idx, h1 = recent_highs[0]
            h2_idx, h2 = recent_highs[1]
            l1_idx, l1 = recent_lows[0]
            l2_idx, l2 = recent_lows[1]
            
            if h2 > h1 and l2 > l1:
                new_status = "up"
            elif h2 < h1 and l2 < l1:
                new_status = "down"
            else:
                new_status = "range"

            if new_status in ["up", "down"]:
                # 추세가 확정되면, 추세를 형성한 첫 번째 Swing Point부터 현재 시점까지 상태를 소급 적용 (Backfill)
                start_idx = min(h1_idx, l1_idx)
                df.loc[start_idx:idx, "status"] = new_status
                current_status = new_status
            else:
                # 추세가 깨진 경우(Range), 직전 Swing Point부터 구간을 Range로 변경하여 시각적 혼란 방지
                prev_swing_idx = l2_idx if h2_idx == idx else h2_idx
                df.loc[prev_swing_idx:idx, "status"] = "range"
                current_status = "range"
        
        df.at[idx, "status"] = current_status

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

def price_action(df):
    df = detect_swing_point(df, fail_limit=5)
    df = calc_trend(df)
    df = detect_reversal_candles(df)

    return df 


def find_reversal_candles(df: pd.DataFrame, lookback=5, pct_threshold=0.02):
    df = df.copy()
    df = df.sort_index(ascending=False)

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 기준 캔들 값
    #━━━━━━━━━━━━━━━━━━━
    base_low  = df["low"].shift(lookback)
    base_high = df["high"].shift(lookback)

    prev_high = df["high"].shift(1)
    prev_low  = df["low"].shift(1)

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 강한 상승 / 하락 (퍼센트 기준)
    #━━━━━━━━━━━━━━━━━━━
    df["strong_up"] = ((prev_high - base_low) / base_low >= pct_threshold)
    df["strong_down"] = ((base_high - prev_low) / base_high >= pct_threshold)

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 캔들 기본 계산
    #━━━━━━━━━━━━━━━━━━━
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 Doji
    #━━━━━━━━━━━━━━━━━━━
    df["doji"] = df["body"] <= df["range"] * 0.1

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 Pin Bar
    #━━━━━━━━━━━━━━━━━━━
    df["bull_pin"] = (df["lower_wick"] >= df["body"] * 2) & (df["upper_wick"] <= df["body"])
    df["bear_pin"] = (df["upper_wick"] >= df["body"] * 2) & (df["lower_wick"] <= df["body"])

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 Inside Bar
    #━━━━━━━━━━━━━━━━━━━
    df["inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 Engulfing
    #━━━━━━━━━━━━━━━━━━━
    df["bull_engulf"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] >= df["open"].shift(1)) &
        (df["open"] <= df["close"].shift(1))
    )

    df["bear_engulf"] = (
        (df["close"] < df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["open"] >= df["close"].shift(1)) &
        (df["close"] <= df["open"].shift(1))
    )

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 최종 시그널
    #━━━━━━━━━━━━━━━━━━━
    df["bull_signal"] = (df["strong_down"] &(df["doji"] | df["bull_pin"] | df["inside_bar"] | df["bull_engulf"]))
    df["bear_signal"] = (df["strong_up"] &(df["doji"] | df["bear_pin"] | df["inside_bar"] | df["bear_engulf"]))

    return df




def calculate_moving_average(df, ma_list):
    #     ma_list = [
    #     {"period": 5},
    #     {"period": 20},
    #     {"period": 60},
    #     {"period": 12, "type": "ema", "name": "fast_ema"},
    # ]
    df = df.copy()
    df = df.sort_index(ascending=False)

    for ma in ma_list:
        period = ma["period"]
        ma_type = ma.get("type", "sma")
        name = ma.get("name", f"{ma_type}_{period}")

        if ma_type == "ema":
            df[name] = df["close"].ewm(span=period, adjust=False).mean()
        else:
            df[name] = df["close"].rolling(window=period).mean()

    return df

def detect_zone(
    df,
    n_pct=0.01,
    min_candles=5,
    max_range_pct=0.005   # ⭐ 추가
):
    """
    n_pct           : 그룹 허용 범위 (예: 0.01 = 1%)
    min_candles     : 매물대 인정 최소 캔들 수
    max_range_pct   : 개별 캔들 변동폭 제한 (예: 0.005 = 0.5%)
    """

    df = df.copy()
    df = df.sort_index(ascending=False)

    zones = []

    group_low = None
    group_high = None
    start_idx = None
    count = 0

    for i in range(len(df)):
        row = df.iloc[i]

        low = row["low"]
        high = row["high"]
        close = row["close"]

        #━━━━━━━━━━━━━━━━━━━
        # 🔹 개별 캔들 변동폭 필터
        #━━━━━━━━━━━━━━━━━━━
        candle_range_pct = (high - low) / close

        if candle_range_pct > max_range_pct:
            # 현재 그룹 종료
            if count >= min_candles:
                zones.append({
                    "start": df.iloc[start_idx].datetime,
                    "end": df.iloc[i - 1].datetime,
                    "low": group_low,
                    "high": group_high,
                    "count": count
                })

            # 그룹 초기화
            group_low = None
            group_high = None
            start_idx = None
            count = 0
            continue

        #━━━━━━━━━━━━━━━━━━━
        # 🔹 그룹 시작
        #━━━━━━━━━━━━━━━━━━━
        if count == 0:
            group_low = low
            group_high = high
            start_idx = i
            count = 1
            continue

        group_mid = (group_high + group_low) / 2
        tolerance = group_mid * n_pct

        in_range = (
            high <= group_mid + tolerance and
            low >= group_mid - tolerance
        )

        if in_range:
            group_low = min(group_low, low)
            group_high = max(group_high, high)
            count += 1
        else:
            # 그룹 종료
            if count >= min_candles:
                zones.append({
                    "start": df.iloc[start_idx].datetime,
                    "end": df.iloc[i - 1].datetime,
                    "low": group_low,
                    "high": group_high,
                    "count": count
                })

            # 새 그룹 시작
            group_low = low
            group_high = high
            start_idx = i
            count = 1

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 마지막 그룹 처리
    #━━━━━━━━━━━━━━━━━━━
    if count >= min_candles:
        zones.append({
            "start": df.iloc[start_idx].datetime,
            "end": df.iloc[i - 1].datetime,
            "low": group_low,
            "high": group_high,
            "count": count
        })

    return zones



def calc_reversal_signals(df):
    df = df.copy()
    df = df.sort_index(ascending=False)
    # #━━━━━━━━━━━━━━━━━━━
    # # 🔹 이동평균
    # #━━━━━━━━━━━━━━━━━━━
    # maFastLen = 5
    # maSlowLen = 20

    # df["maFast"] = df["close"].rolling(maFastLen).mean()
    # df["maSlow"] = df["close"].rolling(maSlowLen).mean()

    # #━━━━━━━━━━━━━━━━━━━
    # # 🔹 추세 판별
    # #━━━━━━━━━━━━━━━━━━━
    # df["upTrend"] = (
    #     (df["maFast"] > df["maSlow"]) &
    #     (df["maFast"] > df["maFast"].shift(1)) &
    #     (df["maSlow"] > df["maSlow"].shift(1))
    # )

    # df["downTrend"] = (
    #     (df["maFast"] < df["maSlow"]) &
    #     (df["maFast"] < df["maFast"].shift(1)) &
    #     (df["maSlow"] < df["maSlow"].shift(1))
    # )

    # df["sideway"] = ~(df["upTrend"] | df["downTrend"])

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 거래량 배율
    #━━━━━━━━━━━━━━━━━━━
    df["volAvg"] = df["volume"].rolling(5).mean()
    df["volRatio"] = df["volume"] / df["volAvg"]

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 캔들 기본 계산
    #━━━━━━━━━━━━━━━━━━━
    df["body"] = (df["close"] - df["open"]).abs()
    df["upperWick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lowerWick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 도지 (정석 정의)
    #━━━━━━━━━━━━━━━━━━━
    df["dojiBase"] = df["body"] <= df["range"] * 0.30

    # 최고/최저 계산 (Pine의 [1] 대응)
    df["highest_6"] = df["high"].rolling(6).max().shift(1)
    df["lowest_6"]  = df["low"].rolling(6).min().shift(1)

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 도지 Exhaustion
    #━━━━━━━━━━━━━━━━━━━
    df["upDojiExhaust"] = (
        # (df["upTrend"] | df["sideway"]) &
        df["dojiBase"] &
        (df["upperWick"] > df["lowerWick"]) &
        (df["high"] >= df["highest_6"])
    )

    df["downDojiExhaust"] = (
        # (df["downTrend"] | df["sideway"]) &
        df["dojiBase"] &
        (df["lowerWick"] > df["upperWick"]) &
        (df["low"] <= df["lowest_6"])
    )

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 장악형 (Engulfing)
    #━━━━━━━━━━━━━━━━━━━
    lowest_5_shift2  = df["low"].rolling(5).min().shift(2)
    highest_5_shift2 = df["high"].rolling(5).max().shift(2)

    df["bullEngulf"] = (
        # (df["downTrend"] | df["sideway"]) &
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] >= df["open"].shift(1)) &
        (df["open"] <= df["close"].shift(1)) &
        (
            (df["low"].shift(1) <= df["lowest_6"]) |
            (df["low"] <= lowest_5_shift2)
        )
    )

    df["bearEngulf"] = (
        # (df["upTrend"] | df["sideway"]) &
        (df["close"] < df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["close"] <= df["open"].shift(1)) &
        (df["open"] >= df["close"].shift(1)) &
        (
            (df["high"].shift(1) >= df["highest_6"]) |
            (df["high"] >= highest_5_shift2)
        )
    )

    #━━━━━━━━━━━━━━━━━━━
    # 🔹 시그널 정리
    #━━━━━━━━━━━━━━━━━━━
    df["signal"] = np.select(
        [
            df["upDojiExhaust"],
            df["bearEngulf"],
            df["downDojiExhaust"],
            df["bullEngulf"]
        ],
        [
            "SELL_DOJI",
            "SELL_ENGULF",
            "BUY_DOJI",
            "BUY_ENGULF"
        ],
        default=None
    )
    
    return df