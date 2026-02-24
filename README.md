# Margin Trading Bot Logic Documentation

이 프로젝트는 암호화폐 선물 거래를 위한 기술적 분석 및 시각화 도구입니다. 핵심 분석 로직은 `logic_manager.py`의 `price_action` 파이프라인을 통해 수행됩니다.

## 📊 Price Action Analysis Logic

`price_action(df)` 함수는 다음 5단계의 순차적인 분석을 통해 시장 상황을 판단합니다.

### 1. Swing Point Detection (`detect_swing_point`)
*   **목적**: 시장의 주요 고점(Swing High)과 저점(Swing Low)을 식별하여 시장 구조를 파악합니다.
*   **로직**:
    *   `fail_limit`(기본 14) 윈도우를 사용하여 국소적 고점/저점을 탐색합니다.
    *   한 캔들 내에서 고점/저점이 동시에 발생할 경우, 캔들의 색상(양봉/음봉)을 기준으로 발생 순서를 논리적으로 정렬합니다.
    *   **Alternation Rule**: High와 Low가 번갈아 나타나도록 필터링하여 노이즈를 제거합니다.

### 2. Trend Calculation (`calc_trend`)
*   **목적**: 식별된 Swing Point를 기반으로 현재 추세(상승, 하락, 횡보)를 결정합니다.
*   **로직**:
    *   최근 2개의 Swing High와 Swing Low를 비교합니다.
    *   **Up Trend (상승)**: 고점이 높아지고(HH) 저점도 높아지는(HL) 경우.
    *   **Down Trend (하락)**: 고점이 낮아지고(LH) 저점도 낮아지는(LL) 경우.
    *   **Range (횡보)**: 위 조건에 해당하지 않는 불명확한 구간.

### 3. Candlestick Pattern Recognition (`detect_reversal_candles`)
*   **목적**: 추세 반전의 잠재적 신호인 캔들 패턴을 탐지합니다.
*   **주요 패턴**:
    *   **Doji**: 시가와 종가가 거의 일치하며 매수/매도 세력이 팽팽한 상태.
    *   **Hammer / Shooting Star**: 긴 꼬리를 가진 반전형 캔들.
    *   **Engulfing (장악형)**: 이전 캔들의 몸통을 완전히 감싸는 강력한 반전 신호.
*   **필터링**: 신뢰도를 높이기 위해 20일 거래량 이동평균(MA)의 1.2배 이상 거래량이 터진 경우만 유효한 패턴으로 인정합니다.

### 4. Volume Profile / Consolidation Zones (`find_volume_profile`)
*   **목적**: 가격이 급격히 변동(Break)하기 전, 에너지가 응축되었던 횡보 구간(매물대)을 찾습니다.
*   **로직**:
    *   가격 변동폭이 급격히 커지는 지점을 감지합니다.
    *   일정 기간(`min_candles=30`) 이상 가격이 특정 범위 내에서 횡보한 구간을 **Zone**으로 설정합니다.

### 5. Trend Zones & Fibonacci Retracement (`find_zone`)
*   **목적**: 추세 구간을 블록화하고, 눌림목(되돌림) 구간을 계산합니다.
*   **로직**:
    *   연속된 추세(Up/Down) 구간을 하나의 그룹으로 묶습니다.
    *   확정된 추세 구간에 대해 **피보나치 되돌림(Fibonacci Retracement)** 레벨을 계산합니다.
        *   주요 레벨: **38.2%, 50%, 61.8%**

---

## 📈 Visualization (`util_manager.py`)
*   `mplfinance`를 사용하여 분석된 데이터를 시각화합니다.
*   Swing Line, 추세 배경색, 매물대 박스, 캔들 패턴 마커, 피보나치 라인 등을 차트에 표시합니다.
*   최신 피보나치 레벨은 차트 우측에 텍스트 라벨로 표시됩니다.