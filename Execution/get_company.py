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


# 여러 기업 키워드 리스트
# company_keywords = {
#     '삼성바이오로직스': '삼성바이오로직스',
#     '셀트리온':'셀트리온',
#     'SK바이오사이언스':'SK바이오사이언스',
#     '셀트리온헬스케어':'셀트리온헬스케어',
#     'SK바이오팜':'SK바이오팜',
#     'HLB':'HLB',
#     '한미약품':'한미약품',
#     '씨젠':'씨젠',
#     '알테오젠':'알테오젠',
#     'SK케미칼':'SK케미칼',
#     '신풍제약':'신풍제약',
#     '메지온':'에지온',
#     '바이오니아':'바이오니아'
# }
# startdate = "20220101"
# enddate = "20221231"
# combined_news = collect_news_for_multiple_keywords(company_keywords, startdate, enddate)
# folder_path = os.getcwd()
# xlsx_file_name = '기업 통합_{}_{}.xlsx'.format(startdate, enddate)
# combined_news.to_excel(xlsx_file_name, index=False)
# print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))
