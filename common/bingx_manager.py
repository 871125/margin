from .config_manager import load_config
import time
import requests
import hmac
import urllib.parse
from hashlib import sha256
from datetime import datetime, timezone, timedelta
import pandas as pd


class BingX :
    def __init__(self):
        config = load_config()

        dictBingx = config['bingx']
        self.apiKey = dictBingx['api_key']
        self.secretKey = dictBingx['secret_key']
        self.apiUrl = dictBingx['api_url']

    def __getSign(self, payload):
        signature = hmac.new(self.secretKey.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
        return signature
    
    def __sendApi(self, method, path, paramsStr, urlParamsStr, payload):
        url = "%s%s?%s&signature=%s" % (self.apiUrl, path, urlParamsStr, self.__getSign(paramsStr))
        headers = {
            'X-BX-APIKEY': self.apiKey,
        }
        response = requests.request(method, url, headers=headers, data=payload)
        return response.json()

    def __parseParam(self, paramsMap):
        sortedKeys = sorted(paramsMap)
        paramsList = []
        urlParamsList = []
        for x in sortedKeys:
            value = paramsMap[x]
            paramsList.append("%s=%s" % (x, value))
        timestamp = str(int(time.time() * 1000))
        paramsStr = "&".join(paramsList)
        if paramsStr != "": 
            paramsStr = paramsStr + "&timestamp=" + timestamp
        else:
            paramsStr = "timestamp=" + timestamp
        contains = '[' in paramsStr or '{' in paramsStr
        for x in sortedKeys:
            value = paramsMap[x]
            if contains:
                encodedValue = urllib.parse.quote(str(value), safe='')
                urlParamsList.append("%s=%s" % (x, encodedValue))
            else:
                urlParamsList.append("%s=%s" % (x, value))
        urlParamsStr = "&".join(urlParamsList)
        if urlParamsStr != "": 
            urlParamsStr = urlParamsStr + "&timestamp=" + timestamp
        else:
            urlParamsStr = "timestamp=" + timestamp
        return paramsStr, urlParamsStr

    def __sendRequest(self, path, method, param):
        # Todo : 오류 처리
        print(">> request start : ", path)
        payload = {}
        paramsStr, urlParamsStr = self.__parseParam(param)
        requestResult = self.__sendApi(method, path, paramsStr, urlParamsStr, payload)

        print(">> request result : " , requestResult['data'])

        return requestResult['data']

    def getOhlcv(self, symbol:str, interval:str, startTime: str, endTime: str):
        """
        get OHLCV data
        :param symbol: Trading pair, for example: BTC-USDT, please use capital letters.
        :param interval: Time interval, refer to field description
        :param startTime: Start time, "2024-01-01 00:00:00"
        :param endTime: End time, "2024-01-01 00:00:00"

        :return: time, open, close, high, low, vol, amount
        """

        # 한국 시간대
        KST = timezone(timedelta(hours=9))
        dtStart = datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S")
        dtStart = dtStart.replace(tzinfo=KST).astimezone(timezone.utc)
        dtStart = dtStart.replace(tzinfo=timezone.utc)
        dtEnd = datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S")
        dtEnd = dtEnd.replace(tzinfo=KST).astimezone(timezone.utc)
        dtEnd = dtEnd.replace(tzinfo=timezone.utc)

        sTime = int(dtStart.timestamp() * 1000)
        eTime = int(dtEnd.timestamp() * 1000)

        paramsMap = {
            "symbol": symbol,
            "interval": interval,
            "limit" : 1440,
            "startTime" : sTime,
            "endTime": eTime
        }
        result = self.__sendRequest(path = '/openApi/swap/v3/quote/klines', method = 'GET', param = paramsMap)
        dfOhlcv = pd.DataFrame(result)
        numericCols = ["open", "close", "high", "low", "volume"]

        dfOhlcv[numericCols] = dfOhlcv[numericCols].astype(float)
        dfOhlcv["datetime_utc"] = pd.to_datetime(dfOhlcv["time"], unit="ms", utc=True)
        dfOhlcv["datetime"] = dfOhlcv["datetime_utc"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        dfOhlcv = dfOhlcv[['datetime', 'open', 'high', 'low', 'close', 'volume']]

        print(">> request result (mod) : " , dfOhlcv)
        return dfOhlcv
    

    def getAccount(self):
        # account
        paramsMap = {}
        accountData = self.__sendRequest(path = '/openApi/swap/v3/user/balance', method = 'GET', param = paramsMap)
        accountData = next((x for x in accountData if x['asset'] == 'USDT'),None)
        
        #   "userId": "116***295", 
        #   "asset": "USDT", : 담보 자산
        #   "balance": "194.8212", : 입금+실현 손익이 반영된 계정 잔액
        #   "equity": "196.7431", : 순자산( balance + unrealizedProfit)
        #   "unrealizedProfit": "1.9219", : 미실현 손익
        #   "realisedProfit": "-109.2504", : 실현 손익(이미 종료된 포지션들의 누적 손익)
        #   "availableMargin": "193.7609", : 사용 가능 마진
        #   "usedMargin": "1.0602", : 사용 중인 마진
        #   "freezedMargin": "0.0000" : 동결 마진(미체결 주문 등 묶인)

        # position
        paramsMap = {}
        positionData = self.__sendRequest(path = '/openApi/swap/v2/user/positions', method = 'GET', param = paramsMap)

        result = {}
        result['account'] = accountData
        result['position'] = positionData


        # "positionId": "1735*****52",
        # "symbol": "BNB-USDT", : 선물 계약 심볼
        # "currency": "USDT", : 마진 통화
        # "positionAmt": "0.20", : 보유중인 포지션 수량
        # "availableAmt": "0.20", : 청산 가능한 수량
        # "positionSide": "SHORT", : 포지션 방향
        # "isolated": true, : 격리마진 여부
        # "avgPrice": "246.43", : 포지션 평단가
        # "initialMargin": "9.7914", : 포지션을 열때 실제로 사용된 증거금
        # "leverage": 5, : 레버리지 배율
        # "unrealizedProfit": "-0.0653", : 미실현 손익
        # "realisedProfit": "-0.0251", : 실현손익
        # "liquidationPrice": 294.16914617776246 : 강제 청산금액

        return result 

    def setLeverage(self, symbol, leverage, positionSide):
        paramsMap = {
            "leverage": leverage,
            "side": positionSide,
            "symbol": symbol
        }
        return self.__sendRequest(path = '/openApi/swap/v2/trade/leverage', method = 'POST', param = paramsMap)
    
    def getLeverage(self, symbol):
        paramsMap = {
            "symbol": symbol
        }
        return self.__sendRequest(path = '/openApi/swap/v2/trade/leverage', method = 'GET', param = paramsMap)
    
    def setMarginType(self, symbol, marginType):
        paramsMap = {
            "symbol": symbol,
            "marginType": marginType,
        }
        return self.__sendRequest(path = '/openApi/swap/v2/trade/marginType', method = 'POST', param = paramsMap)
    
    def getMarginType(self, symbol):
        paramsMap = {
            "symbol": symbol
        }
        return  self.__sendRequest(path = '/openApi/swap/v2/trade/marginType', method = 'GET', param = paramsMap)
    
    def marketBuyOrder(self, symbol, positionSide, quantity, stopLoss=None, takeProfit=None):
        paramsMap = {
            "symbol": symbol,
            "side": "BUY" if positionSide == "LONG" else "SELL",
            "type": "MARKET",
            "positionSide": positionSide, 
            "quantity": quantity
        }
        self.__sendRequest(path = '/openApi/swap/v2/trade/order', method = 'POST', param = paramsMap)
        if stopLoss is not None:
            time.sleep(0.3)
            paramsMap = {
                "symbol": symbol,
                "side": "SELL" if positionSide == "LONG" else "BUY",
                "type": "STOP_MARKET",
                "stopPrice": stopLoss,
                "quantity": quantity,
                "positionSide": positionSide,
                "reduceOnly": True
            }
            self.__sendRequest(path = '/openApi/swap/v2/trade/order', method='POST', param = paramsMap)
        if takeProfit is not None:
            time.sleep(0.3)
            paramsMap = {
                "symbol": symbol,
                "side": "SELL" if positionSide == "LONG" else "BUY",
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": takeProfit,
                "quantity": quantity,
                "positionSide": positionSide,
                "reduceOnly": True
            }
            self.__sendRequest(path = '/openApi/swap/v2/trade/order', method = 'POST', param = paramsMap)
        
        return


    def marketSellOrder(self, symbol, side, positionSide, quantity,):
        paramsMap = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "positionSide": positionSide, # LONG / SHORT
            "quantity": quantity
        }
        return self.__sendRequest(path = '/openApi/swap/v2/trade/order', method = 'POST', param = paramsMap)
    

    