import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# read data
credit = pd.read_csv('credits.csv', encoding='ISO-8859-1')
titles = pd.read_csv('titles.csv', encoding='ISO-8859-1')

# drop na and variable seasons
credit.dropna(inplace=True)
# titles.drop(columns=['seasons'], inplace=True)
columns_to_check = ['imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
titles.dropna(subset=columns_to_check, inplace=True)
titles.to_csv('titles.csv', index=False)

# Delete duplicate value
credit.drop_duplicates(inplace=True)
titles.drop_duplicates(inplace=True)

# change genres type
# titles['genres'] = titles['genres'].apply(ast.literal_eval)

# print(credit.info())
# print(titles.info())


# ------------------------------------------------------------------------------------------------------------


features = ['release_year', 'runtime', 'tmdb_popularity', 'tmdb_score', 'genres', 'production_countries']
target = 'imdb_score'

X = titles[features]
Y = titles[target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=666)

numeric_features = ['release_year', 'runtime', 'tmdb_popularity', 'tmdb_score']
numeric_transformer = SimpleImputer(strategy='median')

categorical_features = ['genres', 'production_countries']
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

model.fit(X_train, Y_train)

# 预测
predictions = model.predict(X_test)

# 计算误差
mse = mean_squared_error(Y_test, predictions)
print(f'Mean Squared Error: {mse}')

# ------------------------------------------------------------------------------------------------------------

# The film is divided into three categories: Short film, Medium-length film and Feature film. And count the number

# Filter out the data whose type is 'movie'
movie_titles = titles[titles['type'] == 'MOVIE'].copy()


# Creates a new column to store the category of the movie
def categorize_runtime(runtime):
    if runtime < 40:
        return 'Short movie'
    elif 40 <= runtime <= 70:
        return 'Medium-length movie'
    else:
        return 'Feature movie'


movie_titles['category'] = movie_titles['runtime'].apply(categorize_runtime)

# Count the number of movies in each category
category_counts = movie_titles['category'].value_counts()

ax = category_counts.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Distribution of Movie Lengths')
plt.xlabel('Category')
plt.ylabel('Number of Movies')
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

plt.show()

# ------------------------------------------------------------------------------------------------------------

# Movies are divided into five categories: G,NC-17,PG, and PG-13. And count the numbers.

# Filter out the data whose type is 'movie'
movie_titles = titles[titles['type'] == 'MOVIE'].copy()

# Filters out the specified five age_certification categories
certifications = ['G', 'NC-17', 'PG', 'PG-13', 'R']
movie_titles = movie_titles[movie_titles['age_certification'].isin(certifications)]

# Count the number of movies in each category
certification_counts = movie_titles['age_certification'].value_counts()

ax = certification_counts.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Distribution of Age Certifications')
plt.xlabel('Certification')
plt.ylabel('Number of Movies')
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

plt.show()


# ------------------------------------------------------------------------------------------------------------

# merge data
merged_data = pd.merge(credit, titles, on='id')

# new csv
merged_data.to_csv('merged_data.csv', index=False)

# ------------------------------------------------------------------------------------------------------------

# Create a graph to show the distribution of IMDb scores

# Group data by IMDb score and count the number of occurrences of each score
imdb_counts = merged_data.groupby('imdb_score').size()

# Create a line plot of IMDb score distribution
plt.figure()
imdb_counts.plot()

# Add title and axis labels
plt.title('IMDb Score Distribution')
plt.xlabel('IMDb Score')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Check out the top 10 actor with the highest average imdb rating

actor_data = merged_data[merged_data['role'] == 'ACTOR']
# Calculate the average imdb score for each character
actor_avg_score = actor_data.groupby('character')['imdb_score'].mean()

# Sort and get the top 10 characters with the highest average rating
top10_actor = actor_avg_score.sort_values(ascending=False).head(10)

print(top10_actor)

color_list = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan']

plt.figure(figsize=(10, 6))
top10_actor.plot(kind='barh', color=color_list)

plt.title('Top 10 Characters(actor) by Average IMDb Score')
plt.xlabel('Average IMDb Score')
plt.ylabel('Character')

for index, value in enumerate(top10_actor):
    plt.text(value, index, f'{value:.2f}')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------------------

# 从credit.csv文件中加载数据
credit_df = pd.read_csv('E:/20008/ASS2/credits.csv')

# 通过对role列进行筛选，提取出role为DIRECTOR的数据
director_data = credit_df[credit_df['role'] == 'DIRECTOR']

# 将筛选后的数据另存为一个新的csv文件，例如directors.csv
director_data.to_csv('directors.csv', index=False)


directors_df = pd.read_csv('directors.csv')
titles_df = pd.read_csv('titles.csv')

# 根据id列合并两个数据集
merged_df = pd.merge(directors_df, titles_df, on='id')

# 确保合并后的数据集中的角色是DIRECTOR（虽然directors数据集中已经是DIRECTOR，但是为了保险起见，我们再次检查）
merged_directors_df = merged_df[merged_df['role'] == 'DIRECTOR']

# 按character列分组，并计算每组的平均imdb_score
character_avg_score = merged_directors_df.groupby('character')['imdb_score'].mean()

# 根据平均imdb_score降序排列，取前10
top10_characters = character_avg_score.sort_values(ascending=False).head(10)

# 输出结果
print(top10_characters)

top10_characters.plot(kind='bar', figsize=(10, 6))

# 添加标题和标签
plt.title('Top 10 Characters by Average Imdb Score')
plt.xlabel('Character')
plt.ylabel('Average Imdb Score')

# 显示图表
plt.show()