import pandas as pd

credit = pd.read_csv('E:/20008/ASS2/credits.csv')
title = pd.read_csv('E:/20008/ASS2/titles.csv')

print(credit.info())
print(title.info())

# drop na
credit.dropna(inplace=True)
columns_to_check = ['imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
title.dropna(subset=columns_to_check, inplace=True)
title.to_csv('E:/20008/ASS2/titles.csv', index=False)

# Delete duplicate value
credit.drop_duplicates(inplace=True)
title.drop_duplicates(inplace=True)

# Sort the roles in the credits file
sorted_credits = credit.sort_values(by='role', ascending=True)






































































