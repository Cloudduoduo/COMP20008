import pandas as pd
import pandas as ps

credits = pd.read_csv('E:/20008/ASS2/credits.csv')
titles = pd.read_csv('E:/20008/ASS2/titles.csv')

print(credits.info())
print(titles.info())

# drop na
credits.dropna(inplace=True)
titles.dropna(inplace=True)

# Delete duplicate value
credits.drop_duplicates(inplace=True)
titles.drop_duplicates(inplace=True)
# Sort the roles in the credits file
sorted_credits = credits.sort_values(by='role', ascending=True)






































































