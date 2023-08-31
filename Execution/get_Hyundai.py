from Crawling import crawling
import os

keywords = '현대차'
startdate = "20180101"
enddate = "20180131"

news = crawling.crawling_get_news(keywords, startdate, enddate)

folder_path = os.getcwd()

xlsx_file_name = '{}_{}_{}.xlsx'.format(keywords, startdate, enddate)

news.to_excel(xlsx_file_name, index=False)

print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))
