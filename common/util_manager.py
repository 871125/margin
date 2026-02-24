
from .bingx_manager import BingX
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, df, title):
        self.title = title
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        self.df = df.sort_index()

    def candle_price_action(self, volume=True):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Price Action Comprehensive Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. Swing Line & Trend Background (Up/Down/Range)
        # 2. Dynamic Trend Zones (Support/Resistance)
        # 3. Latest Fibonacci Retracement Levels (38.2%, 50%, 61.8%)
        # 4. Candlestick Patterns (Doji, Hammer, Engulfing)
        add_plots = []

        # ─────────────────────────────
        # 🔹 swing 라인
        # ─────────────────────────────
        self.df["swing"] = self.df["swing_high"].combine_first(self.df["swing_low"])
        self.df["swing_line"] = self.df["swing"].interpolate(method="time", limit_area='inside')

        add_plots.append(
            mpf.make_addplot(
                self.df["swing_line"].values,
                type="line",
                width=1.5,
                label="swing point"
            )
        )

        # numpy 변환
        ymin = self.df["low"].min()
        ymax = self.df["high"].max()

        ymin_arr = [ymin] * len(self.df)
        ymax_arr = [ymax] * len(self.df)

        fill_between = [
            # 상승
            dict(
                y1=ymin_arr,
                y2=ymax_arr,
                where=(self.df["status"] == "up").to_numpy(),
                color="blue",
                alpha=0.5
            ),
            # 하락
            dict(
                y1=ymin_arr,
                y2=ymax_arr,
                where=(self.df["status"] == "down").to_numpy(),
                color="red",
                alpha=0.5
            ),
            # 횡보
            dict(
                y1=ymin_arr,
                y2=ymax_arr,
                where=(self.df["status"] == "range").to_numpy(),
                color="gold",
                alpha=0.5
            ),
        ]

        if "zone_high" in self.df.columns and "zone_low" in self.df.columns:
            fill_between.append(
                dict(
                    y1=self.df["zone_low"].fillna(0).values,
                    y2=self.df["zone_high"].fillna(0).values,
                    where=self.df["zone_id"].notna().values,
                    color="purple",
                    alpha=0
                )
            )

        # Dynamic zone plotting
        up_zone_cols = [c for c in self.df.columns if c.startswith("trend_up_zone_") and c.endswith("_high")]
        down_zone_cols = [c for c in self.df.columns if c.startswith("trend_down_zone_") and c.endswith("_high")]

        for col in up_zone_cols:
            low_col = col.replace("_high", "_low")
            if low_col in self.df.columns:
                fill_between.append(
                    dict(
                        y1=self.df[low_col].values,
                        y2=self.df[col].values,
                        where=self.df[low_col].notna().values,
                        color="green",
                        alpha=0.1
                    )
                )

        for col in down_zone_cols:
            low_col = col.replace("_high", "_low")
            if low_col in self.df.columns:
                fill_between.append(
                    dict(
                        y1=self.df[low_col].values,
                        y2=self.df[col].values,
                        where=self.df[low_col].notna().values,
                        color="red",
                        alpha=0.1
                    )
                )

        fib_cols = [c for c in self.df.columns if "_fib_" in c]
        
        # ─────────────────────────────────────────────────────────────
        # 🔹 마지막 시점(Current)의 피보나치만 남기기
        # ─────────────────────────────────────────────────────────────
        current_idx = self.df.index[-1]
        final_fib_series = {}
        
        for col in fib_cols:
            # 현재 시점에 값이 있는 경우에만 포함
            if pd.notna(self.df.loc[current_idx, col]):
                series = self.df[col].copy()
                # 현재 블록만 남기고 이전 데이터(과거 Zone) 제거
                if series.isna().any():
                    last_nan_idx = series.isna()[::-1].idxmax()
                    series.loc[:last_nan_idx] = np.nan
                final_fib_series[col] = series

        for col, series in final_fib_series.items():
            is_50 = "_500" in col
            is_up = "trend_up" in col
            
            if is_up:
                color = "blue" if is_50 else "skyblue"
            else:
                color = "red" if is_50 else "lightcoral"

            width = 1.2 if is_50 else 0.8

            add_plots.append(
                mpf.make_addplot(
                    series,
                    type='line',
                    linestyle='dotted',
                    width=width,
                    color=color
                )
            )

        bear_doji_y = np.where(self.df["bearDoji"], self.df["high"] * 1.002, np.nan)
        if np.any(~np.isnan(bear_doji_y)):
            add_plots.append(
                mpf.make_addplot(
                    bear_doji_y,
                    type="scatter",
                    marker="o",
                    markersize=30,
                    color="blue"
                )
            )

        bull_doji_y = np.where(self.df["bullDoji"], self.df["low"] * 0.998, np.nan)
        if np.any(~np.isnan(bull_doji_y)):
            add_plots.append(
                mpf.make_addplot(
                    bull_doji_y,
                    type="scatter",
                    marker="o",
                    markersize=30,
                    color="red"
                )
            )

        bear_hs_y = np.where(self.df["bearHS"], self.df["high"] * 1.002, np.nan)
        if np.any(~np.isnan(bear_hs_y)):
            add_plots.append(
                mpf.make_addplot(
                    bear_hs_y,
                    type="scatter",
                    marker="v",
                    markersize=30,
                    color="blue"
                )
            )

        bull_hs_y = np.where(self.df["bullHS"], self.df["low"] * 0.998, np.nan)
        if np.any(~np.isnan(bull_hs_y)):
            add_plots.append(
                mpf.make_addplot(
                    bull_hs_y,
                    type="scatter",
                    marker="^",
                    markersize=30,
                    color="red"
                )
            )


        bear_engulfing_y = np.where(self.df["bearEngulf"], self.df["high"] * 1.002, np.nan)
        if np.any(~np.isnan(bear_engulfing_y)):
            add_plots.append(
                mpf.make_addplot(
                    bear_engulfing_y,
                    type="scatter",
                    marker="*",
                    markersize=30,
                    color="blue"
                )
            )

        bull_engulfing_y = np.where(self.df["bullEngulf"], self.df["low"] * 0.998, np.nan)
        if np.any(~np.isnan(bull_engulfing_y)):
            add_plots.append(
                mpf.make_addplot(
                    bull_engulfing_y,
                    type="scatter",
                    marker="*",
                    markersize=30,
                    color="red"
                )
            )

        fig, axes = mpf.plot(
            self.df,
            type="candle",
            volume=volume,
            addplot=add_plots,
            fill_between=fill_between,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8),
            returnfig=True
        )

        # 텍스트 라벨 추가 (피보나치 % 표시)
        ax = axes[0]
        for col, series in final_fib_series.items():
            last_idx = series.last_valid_index()
            if last_idx:
                val = series.loc[last_idx]
                x_pos = self.df.index.get_loc(last_idx)
                
                label = ""
                if "_382" in col: label = "38.2%"
                elif "_500" in col: label = "50%"
                elif "_618" in col: label = "61.8%"
                
                is_50 = "_500" in col
                is_up = "trend_up" in col
                
                if is_up:
                    text_color = "blue" if is_50 else "skyblue"
                else:
                    text_color = "red" if is_50 else "lightcoral"

                ax.text(x_pos, val, f" {label}", color=text_color, fontsize=9, verticalalignment="center", fontweight='bold' if is_50 else 'normal')

        plt.show()
        return fig, axes

    def candle_signal(self, volume=True):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Candle Pattern Signal Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. 주요 캔들 패턴(Doji, Hammer, Engulfing) 발생 지점에 마커 표시
        df = self.df[["open", "high", "low", "close", "volume"]].copy()

        apds = []


        bear_doji_y = np.where(self.df["bearDoji"], self.df["high"] * 1.002, np.nan)
        if np.any(~np.isnan(bear_doji_y)):
            apds.append(
                mpf.make_addplot(
                    bear_doji_y,
                    type="scatter",
                    marker="o",
                    markersize=30,
                    color="blue"
                )
            )

        bull_doji_y = np.where(self.df["bullDoji"], self.df["low"] * 0.998, np.nan)
        if np.any(~np.isnan(bull_doji_y)):
            apds.append(
                mpf.make_addplot(
                    bull_doji_y,
                    type="scatter",
                    marker="o",
                    markersize=30,
                    color="red"
                )
            )

        bear_hs_y = np.where(self.df["bearHS"], self.df["high"] * 1.002, np.nan)
        if np.any(~np.isnan(bear_hs_y)):
            apds.append(
                mpf.make_addplot(
                    bear_hs_y,
                    type="scatter",
                    marker="v",
                    markersize=30,
                    color="blue"
                )
            )

        bull_hs_y = np.where(self.df["bullHS"], self.df["low"] * 0.998, np.nan)
        if np.any(~np.isnan(bull_hs_y)):
            apds.append(
                mpf.make_addplot(
                    bull_hs_y,
                    type="scatter",
                    marker="^",
                    markersize=30,
                    color="red"
                )
            )


        bear_engulfing_y = np.where(self.df["bearEngulf"], self.df["high"] * 1.002, np.nan)
        if np.any(~np.isnan(bear_engulfing_y)):
            apds.append(
                mpf.make_addplot(
                    bear_engulfing_y,
                    type="scatter",
                    marker="*",
                    markersize=30,
                    color="blue"
                )
            )

        bull_engulfing_y = np.where(self.df["bullEngulf"], self.df["low"] * 0.998, np.nan)
        if np.any(~np.isnan(bull_engulfing_y)):
            apds.append(
                mpf.make_addplot(
                    bull_engulfing_y,
                    type="scatter",
                    marker="*",
                    markersize=30,
                    color="red"
                )
            )

        return mpf.plot(
            df,
            type="candle",
            volume=volume,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8),
            addplot=apds
        )


    def candle(self, volume):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Basic Candle Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. 기본 캔들 차트 및 거래량 표시
        self.df = self.df[["open", "high", "low", "close", "volume"]]
        return mpf.plot(
            self.df,
            type="candle",
            volume=volume,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8)
        )
    
    def candle_ma(self, lstMa, volume):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Moving Average Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. 다중 이동평균선(MA) 표시
        add_plots =[]
        for ma in lstMa:
            period = ma["period"]
            maType = ma.get("type", "sma")
            name = ma.get("name", f"{maType}_{period}")
            
            add_plots.append(
                mpf.make_addplot(
                    self.df[name],
                    type="line",
                    width=1.5,
                    label=name
                )
            )

        return mpf.plot(
            self.df,
            type="candle",
            volume=volume,
            addplot=add_plots,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8)
        )

    def candle_swing(self, volume):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Swing & Trend Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. Swing Point 연결선 표시
        # 2. 추세 상태(Up/Down/Range)에 따른 배경색 채우기
        self.df = self.df[["open", "high", "low", "close", "volume", "swing_high", "swing_low", "status"]]
        add_plots = []

        # ─────────────────────────────
        # 🔹 swing 라인
        # ─────────────────────────────
        self.df["swing"] = self.df["swing_high"].combine_first(self.df["swing_low"])
        self.df["swing_line"] = self.df["swing"].interpolate(method="time")

        add_plots.append(
            mpf.make_addplot(
                self.df["swing_line"].values,
                type="line",
                width=1.5,
                label="swing point"
            )
        )

        # numpy 변환
        ymin = self.df["low"].min()
        ymax = self.df["high"].max()

        ymin_arr = [ymin] * len(self.df)
        ymax_arr = [ymax] * len(self.df)

        fill_between = [
            # 상승
            dict(
                y1=ymin_arr,
                y2=ymax_arr,
                where=(self.df["status"] == "up").to_numpy(),
                color="blue",
                alpha=0.06
            ),
            # 하락
            dict(
                y1=ymin_arr,
                y2=ymax_arr,
                where=(self.df["status"] == "down").to_numpy(),
                color="red",
                alpha=0.06
            ),
            # 횡보
            dict(
                y1=ymin_arr,
                y2=ymax_arr,
                where=(self.df["status"] == "range").to_numpy(),
                color="gold",
                alpha=0.06
            ),
        ]

        return mpf.plot(
            self.df,
            type="candle",
            volume=volume,
            addplot=add_plots,
            fill_between=fill_between,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8)
        )


        
    def candle_swing_reversal(self, volume):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Swing Reversal Signal Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. 매매 신호(BUY/SELL) 발생 지점 마커 표시
        # 2. Swing Line 함께 표시
        df = self.df.copy()   # 🔥 self.df 건드리지 마라

        add_plots = []

        # 🔺 상승 반전 (NaN 방식)
        rev_up = np.where(
                    df["signal"].str.contains("BUY_", na=False),
                    df["low"],
                    np.nan
                )
        add_plots.append(
            mpf.make_addplot(
                rev_up,
                type="scatter",
                marker="^",
                markersize=80
            )
        )

        # 🔻 하락 반전
        rev_down = np.where(\
                    df["signal"].str.contains("SELL_", na=False),
                    df["high"],
                    np.nan
                )
        add_plots.append(
            mpf.make_addplot(
                rev_down,
                type="scatter",
                marker="v",
                markersize=80
            )
        )

        # 🔹 swing line
        self.df["swing"] = self.df["swing_high"].combine_first(self.df["swing_low"])
        self.df["swing_line"] = self.df["swing"].interpolate(method="time")
        add_plots.append(
            mpf.make_addplot(
                self.df["swing_line"],
                type="line",
                width=1.5,
                label='swing point'
            )
        )

        return mpf.plot(
            df[["open", "high", "low", "close", "volume"]],
            type="candle",
            volume=volume,
            addplot=add_plots,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8)
        )
        
    def candle_zone(self, volume=True, zones=None):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔹 Supply & Demand Zone Chart
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. 특정 매물대(Zone)를 박스 형태로 시각화
        self.df = self.df[["open", "high", "low", "close", "volume"]]

        fig, axes = mpf.plot(
            self.df,
            type="candle",
            volume=volume,
            style="charles",
            title=self.title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8),
            returnfig=True
        )

        ax = axes[0]  # 가격 차트

        #━━━━━━━━━━━━━━━━━━━
        # 🔹 매물대 박스 그리기
        #━━━━━━━━━━━━━━━━━━━
        if zones:
            for z in zones:
                x_start = self.df.index.get_indexer([z["start"]], method="nearest")[0]
                x_end   = self.df.index.get_indexer([z["end"]],   method="nearest")[0]

                width = max(x_end - x_start, 1)

                rect = patches.Rectangle(
                    (x_start, z["low"]),
                    width,
                    z["high"] - z["low"],
                    linewidth=1,
                    edgecolor="purple",
                    facecolor="purple",
                    alpha=0.25
                )

                ax.add_patch(rect)
        plt.show()
        return fig, axes