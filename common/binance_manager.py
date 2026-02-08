import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from .config_manager import load_config

class Binance:
    def __init__(self):
        # config.yaml에 binance 섹션이 있다고 가정합니다.
        # 공용 데이터(OHLCV) 조회에는 API Key가 필수는 아니지만, 클래스 구조 유지를 위해 로드합니다.
        try:
            config = load_config()
            dictBinance = config.get('binance', {})
            self.apiUrl = dictBinance.get('api_url', 'https://fapi.binance.com') # 기본값은 선물 API
        except Exception:
            self.apiUrl = 'https://fapi.binance.com'

    def getOhlcv(self, symbol: str, interval: str, startTime: str, endTime: str):
        """
        바이낸스 선물 OHLCV 데이터를 가져옵니다.
        :param symbol: 예: 'BTCUSDT' (바이낸스는 하이픈 없이 사용)
        :param interval: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'
        :param startTime: "2024-01-01 00:00:00"
        :param endTime: "2024-01-01 00:00:00"
        """
        # 하이픈 제거 (BTC-USDT -> BTCUSDT)
        symbol = symbol.replace('-', '')

        # 한국 시간대 처리
        KST = timezone(timedelta(hours=9))
        dtStart = datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        dtEnd = datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)

        sTime = int(dtStart.timestamp() * 1000)
        eTime = int(dtEnd.timestamp() * 1000)

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": sTime,
            "endTime": eTime,
            "limit": 1500
        }

        path = "/fapi/v1/klines"
        url = self.apiUrl + path
        
        response = requests.get(url, params=params)
        data = response.json()

        if isinstance(data, dict) and "code" in data:
            print(f">> Binance API Error: {data['msg']}")
            return pd.DataFrame()

        # 데이터 프레임 생성
        df = pd.DataFrame(data, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # 필요한 컬럼 추출 및 타입 변환
        numericCols = ["open", "high", "low", "close", "volume"]
        df[numericCols] = df[numericCols].astype(float)
        
        # 시간대 변환 (UTC -> Asia/Seoul)
        df["datetime_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["datetime"] = df["datetime_utc"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f">> Binance Request Success: {symbol} ({len(df)} rows)")
        return df