from common.slack_manager import Slack
from common.config_manager import load_config
from common.bingx_manager import BingX
from common.binance_manager import Binance


from common.util_manager import Graph
from common.logic_manager import price_action
import pandas as pd

# from common.logic_manager import detect_supply_demand

if __name__ == "__main__":
    clsBingX = BingX()
    clsBinance = Binance()
    df = clsBinance.getOhlcv('BTC-USDT', '4h', '2025-01-01 00:00:00', '2026-02-24  00:00:00')
    df = price_action(df)

    g = Graph(df, "BTC-USDT")
    g.candle_price_action(volume = True)


    t = clsBingX.getAccount()
    print(t)


    # availableLongVol * safe_ratio
    
    # availableShortVol * safe_ratio
    clsGraph = Graph(df, 'BTC-USDT')
    clsGraph.candle(True)
    print('t')


    # config = load_config()
    # clsSlack = Slack()

    # clsSlack.message('test')
