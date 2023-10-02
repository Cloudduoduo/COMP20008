import pandas as pd

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
