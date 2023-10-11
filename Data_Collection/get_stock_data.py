#주가 데이터 가져오기 클래스화
import FinanceDataReader as fdr
import os
import pandas as pd
current_directory = os.getcwd() # 현재 스크립트 파일의 경로를 가져오기
parent_directory = os.path.dirname(current_directory) # 현재 스크립트 파일의 디렉토리 경로 (상위 디렉토리)를 계산
print(parent_directory)
print(current_directory)
# 코스피와 코스닥 데이터 가져오기
kospi_data = fdr.StockListing("KOSPI")
kosdaq_data = fdr.StockListing("KOSDAQ")

#코드로 종목 가져오기 때문에 종목과 코드를 매핑한 딕셔너리 생성 
stock_mapping = {}
for _, row in kospi_data.iterrows():
    stock_mapping[row["Name"]] = row["Code"]
for _, row in kosdaq_data.iterrows():
    stock_mapping[row["Name"]] = row["Code"]

# 주가와 거래량 데이터 가져오기 클래스화
class stock_data():
    def __init__(self, sector_name, stock_list, stock_mapping):   #생성 시 섹터의 종목리스트(stock_list)와 종목코드와 종목명을 매핑한 stock_mapping을 인자로 받는다.    
        self.sector_name = sector_name
        self.stock_list = stock_list
        self.stock_mapping = stock_mapping        
        self.data_frames = {}                        #차후 stock_list의 모든 종목의 주가데이터를 하나로 합치기 위해 딕셔너리를 하나 생성해둔다.

    def get_stock_data(self,start_date, end_date):   #주가 데이터를 가져오는 코드
        for stock in self.stock_list:
            data = fdr.DataReader(stock_mapping[stock], start_date, end_date).reset_index()
            data['stock'] = stock                    #주식명을 컬럼으로 넣어주기
            setattr(self, stock, data)               #주가 데이터의 변수명을 해당 주식으로 하게끔하기 위함
            self.data_frames[stock] = data           
    
    def get_stock_list(self):                        #어떤 종목들이 인스턴스에 있는지 확인할 수 있음
        print(self.stock_list)
        
    def get_all_dataframe(self):                     #클래스 인스턴스 내의 모든 인스턴스들을 하나의 데이터프레임으로 합침
        self.combined_data = pd.concat(list(self.data_frames.values()), ignore_index=True)
        # setattr(self, f"{self.stock_list}_combined_data", pd.concat(list(self.data_frames.values()), ignore_index=True))
    
    def file_save(self):                             #지정된 경로(data디렉토리)로 수집한 데이터를 저장
        self.combined_data.to_csv(f"{current_directory}/data/{self.sector_name}_combined.csv")
    
    def add_stock(self, stock):
        self.stock_list.append(stock)
        
bio = ['삼성바이오로직스','셀트리온','SK바이오사이언스','SK바이오팜','HLB','한미약품','씨젠','알테오젠','SK케미칼','신풍제약','메지온','바이오니아']
semiconductor = ['삼성전자','한미반도체','DB하이텍','주성엔지니어링','원익IPS','하나마이크론','KEC']
car =['현대차','기아','KG모빌리티']

start_date = '2004-01-01'
end_date = '2023-08-31'

# bio_stocks = stock_data('bio', bio, stock_mapping)
# bio_stocks.get_stock_data(start_date,end_date)
# bio_stocks.get_stock_list()
# bio_stocks.get_all_dataframe()
# bio_stocks.file_save()

car_stocks = stock_data('car', car, stock_mapping)
car_stocks.get_stock_data(start_date,end_date)
car_stocks.get_stock_list()
car_stocks.get_all_dataframe()
car_stocks.file_save()

# semiconductor_stocks = stock_data('semiconductor', semiconductor, stock_mapping)
# semiconductor_stocks.get_stock_data(start_date,end_date)
# semiconductor_stocks.get_stock_list()
# semiconductor_stocks.get_all_dataframe()
# semiconductor_stocks.file_save()

# for i, stock in enumerate(bio):
#     # 종목 데이터를 가져오고 reset_index를 적용
#     data = fdr.DataReader(stock_mapping[stock], start_date, end_date).reset_index()
#     data['stock'] = stock
#     # 변수를 동적으로 생성
#     globals()[stock] = data    
#     # 데이터프레임을 리스트에 추가
#     bio_list.append(globals()[stock])

# bio_list
