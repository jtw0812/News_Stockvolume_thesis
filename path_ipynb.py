import os
import sys

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# 현재 디렉토리 내의 모든 하위 디렉토리를 가져옴
subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
# print(current_directory)
# 각 하위 디렉토리에 대해 경로를 추가
for subdir in subdirectories:
    subdir_path = os.path.join(parent_directory, subdir)
    sys.path.append(subdir_path)

# 모듈 검색 경로에 현재 디렉토리도 추가
sys.path.append(parent_directory)

# sys.path를 출력하여 추가된 경로 확인
print(sys.path)
# print(current_directory)