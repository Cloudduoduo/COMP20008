import pandas as pd

credit = pd.read_csv('E:/20008/ASS2/credits.csv')
title = pd.read_csv('E:/20008/ASS2/titles.csv')

print(credit.info())
print(title.info())

# drop na
credit.dropna(inplace=True)
title.dropna(inplace=True)

# Delete duplicate value
credit.drop_duplicates(inplace=True)
title.drop_duplicates(inplace=True)
# Sort the roles in the credits file
sorted_credits = credit.sort_values(by='role', ascending=True)

merged_data = pd.merge(credit, title, on='id')
merged_data.to_csv('merged_data.csv', index=False)



































































