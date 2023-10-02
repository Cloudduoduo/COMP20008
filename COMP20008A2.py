import pandas as pd
import ast


# read data
credit = pd.read_csv('credits.csv', encoding='ISO-8859-1')
titles = pd.read_csv('titles.csv', encoding='ISO-8859-1')

# print(credit.info())
# print(titles.info())

# drop na
credit.dropna(inplace=True)
titles.dropna(inplace=True)

# Delete duplicate value
credit.drop_duplicates(inplace=True)
titles.drop_duplicates(inplace=True)


# ------------------------------------------------------------------------------------------------------------

all_genres = []

# 遍历dataframe中的每一行
for index, row in titles.iterrows():
    # 将字符串转换为列表
    genres_list = ast.literal_eval(row['genres'])

    # 将这一行中的每个genre添加到all_genres列表中
    all_genres.extend(genres_list)

# 使用set来找到all_genres列表中唯一的genres
unique_genres = set(all_genres)

# 打印出结果
print("Number of unique genres:", len(unique_genres))
print("List of unique genres:", unique_genres)