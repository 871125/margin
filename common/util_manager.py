
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
        # self.df = self.df[["open", "high", "low", "close", "volume", "swing_high", "swing_low", "status"]]
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
                    alpha=0.3
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

    def candle_signal(self, volume=True):
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