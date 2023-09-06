import pandas as pd


def process_stock_data(df):
    unique_stocks = df['stock'].unique()  # 주식명(stock)의 고유한 값들을 가져옴    
    processed_dfs = []  # 처리된 데이터프레임들을 저장할 리스트
    
    for stock_name in unique_stocks:
        stock_df = df[df['stock'] == stock_name].copy()  # 주식명에 해당하는 부분을 선택하여 복사
        
        ### 주가 상승 예측 모델 생성 위한 1일 1주일 1달 3달 6달 1년 뒤 상승여부 컬럼 만들기(종가 기준)
        stock_df['1day_updown'] = (stock_df['Close'].shift(-1) > stock_df['Close']).astype(int)
        stock_df['1week_updown'] = (stock_df['Close'].shift(-5) > stock_df['Close']).astype(int)
        stock_df['1month_updown'] = (stock_df['Close'].shift(-20) > stock_df['Close']).astype(int)  # 가정: 1달은 20 영업일로 계산
        stock_df['3month_updown'] = (stock_df['Close'].shift(-60) > stock_df['Close']).astype(int)  # 가정: 3달은 60 영업일로 계산
        stock_df['6month_updown'] = (stock_df['Close'].shift(-120) > stock_df['Close']).astype(int)  # 가정: 6달은 120 영업일로 계산
        stock_df['1year_updown'] = (stock_df['Close'].shift(-240) > stock_df['Close']).astype(int)
        ### 논문의 거래량 조정 방법인 100거래일 최저 거래량 빼주기(0이 안 나오게끔 0.9를 곱해서 보정 해주었음)
        # minus = stock_df['Volume'].rolling(window=100).min()
        # minus.fillna(method='bfill', inplace=True) # 뒤의 값을 통해서 결측값을 채우기(첫 99일 동안은 nan이 나오기 때문)
        # minus = minus * 0.1
        # stock_df['Target'] = stock_df['Volume'] - minus
        stock_df['Target'] = stock_df['Volume']
        ## 주말 공휴일 등, 거래되지 않은 날의 주가 및 거래 데이터를 앞의 날의 정보로 채워 넣기
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        all_dates = pd.date_range(start=min(stock_df['Date']), end=max(stock_df['Date']))  # 모든 날짜 생성
        stock_df = stock_df.set_index('Date').reindex(all_dates)  # 날짜 인덱스 재설정 및 빈 날짜 생성
        stock_df = stock_df.fillna(method='ffill')  # 빈 값을 앞쪽으로 채우기
                  
       
        processed_dfs.append(stock_df)
    
    # 모든 데이터프레임을 병합
    result_df = pd.concat(processed_dfs, axis=0)
    result_df = result_df.reset_index()
    result_df = result_df.rename(columns={'index': 'Date'})

    # 예측을 위한 수정 거래량 구하는 코드   

    return result_df

def get_analyze_df(text_data, stock_data):
    stock_df = process_stock_data(stock_data)
    return pd.merge(text_data, stock_df , how = 'left', on =['stock','Date'])