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
    df = df.sort_index(ascending=False)
    df["swing_high"] = np.nan
    df["swing_low"]  = np.nan

    n = len(df)
    start = 0
    isHigh = True

    while start < n - 1:
        failCnt = 0

        if isHigh:
            highVal = df.iloc[start].high
            highIdx = start

            for i in range(start + 1, n):
                if df.iloc[i].high >= highVal:
                    highVal = df.iloc[i].high
                    highIdx = i
                    failCnt = 0
                else:
                    failCnt += 1

                if failCnt >= fail_limit:
                    df.iat[highIdx, df.columns.get_loc("swing_high")] = highVal
                    # 🔥 핵심: 무조건 앞으로
                    start = highIdx + 1
                    isHigh = False
                    break
            else:
                break

        else:
            lowVal = df.iloc[start].low
            lowIdx = start

            for i in range(start + 1, n):
                if df.iloc[i].low <= lowVal:
                    lowVal = df.iloc[i].low
                    lowIdx = i
                    failCnt = 0
                else:
                    failCnt += 1

                if failCnt >= fail_limit:
                    df.iat[lowIdx, df.columns.get_loc("swing_low")] = lowVal
                    # 🔥 핵심: 무조건 앞으로
                    start = lowIdx + 1
                    isHigh = True
                    break
            else:
                break
    if isHigh:
        df.at[df.index[-1], 'swing_high'] = df.at[df.index[-1], 'high']
    else:
        df.at[df.index[-1], 'swing_low'] = df.at[df.index[-1], 'low']
    return df

def calc_trend(df):
    df = df.copy()
    df["status"] = "unknown"
    first = True

    lstSwingHigh = []   # [(i, value)]
    lstSwingLow  = []   # [(i, value)]

    last_swing_i = None  # 🔑 직전 swing index

    for i in range(len(df)):
        idx = df.index[i]
        sh = df.at[idx, "swing_high"]
        sl = df.at[idx, "swing_low"]

        is_swing = False

        # ───────── swing high ─────────
        if not np.isnan(sh):
            lstSwingHigh.append((i, sh))
            if len(lstSwingHigh) > 2:
                lstSwingHigh = lstSwingHigh[1:]
            is_swing = True

        # ───────── swing low ─────────
        if not np.isnan(sl):
            lstSwingLow.append((i, sl))
            if len(lstSwingLow) > 2:
                lstSwingLow = lstSwingLow[1:]
            is_swing = True

        # swing이 아닌 봉은 스킵
        if not is_swing:
            continue

        # 4개 스윙 확보되었을 때만 판단
        if len(lstSwingHigh) == 2 and len(lstSwingLow) == 2:
            h1, h2 = lstSwingHigh[0][1], lstSwingHigh[1][1]
            l1, l2 = lstSwingLow[0][1], lstSwingLow[1][1]
            indices = [
                lstSwingHigh[0][0],
                lstSwingHigh[1][0],
                lstSwingLow[0][0],
                lstSwingLow[1][0],
            ]


            if h1 < h2 and l1 < l2:
                status = "up"
            elif h1 > h2 and l1 > l2:
                status = "down"
            else:
                status = "range"

            # # 🔑 핵심 변경 부분
            # start_i = last_swing_i if last_swing_i is not None else i
            # end_i = i

            if first:
                df.iloc[min(indices) : max(indices) + 1, df.columns.get_loc("status")] = status
                first = False
            else:
                df.iloc[sorted(indices)[2] : max(indices) + 1, df.columns.get_loc("status")] = status


            # df.iloc[start_i : end_i, df.columns.get_loc("status")] = status

        # 현재 swing을 마지막 swing으로 저장
        last_swing_i = i

    return df

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