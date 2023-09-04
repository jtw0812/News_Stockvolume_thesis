import sys, os
import importlib
current_directory = os.getcwd()
project_directory = os.path.dirname(current_directory)
print(project_directory)
print(current_directory)
sys.path.append(project_directory)
sys.path.append(current_directory)

import Data_Collection
import LASSO
import path
import eKoNL
# print(path.current_directory)

# 현재 스크립트가 위치한 디렉토리의 경로를 계산

# eKoNL.py 파일이 위치한 디렉토리와 Execution 디렉토리의 경로를 계산


# 해당 디렉토리들을 모듈 검색 경로에 추가


# import crawling

# keywords = '현대차'
# startdate = "20180101"
# enddate = "20180131"

# news = crawling.crawling_get_news(keywords, startdate, enddate)

# folder_path = os.path.join(execution_directory, "data")

# xlsx_file_name = '{}_{}_{}.xlsx'.format(keywords, startdate, enddate)

# news.to_excel(xlsx_file_name, index=False)

# print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))
