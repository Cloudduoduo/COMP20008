import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

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

# print(credit.info())
# print(titles.info())


# regression------------------------------------------------------------------------------------------------------------


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

# 饼图------------------------------------------------------------------------------------------------------------

# change genres type
pieChartDF = titles
# Assuming pieChartDF = titles, ensure the genres column is a list.
pieChartDF['genres'] = pieChartDF['genres'].apply(ast.literal_eval)

# Explode the genres column into individual rows
p_exploded = pieChartDF['genres'].explode()

# Calculate the genre counts
genre_counts = p_exploded.value_counts()
total = genre_counts.sum()

# Define a threshold, e.g., 2%
threshold = 0.02 * total

# Create a new Series object where smaller categories are grouped as "Other"
genre_counts_adjusted = genre_counts[genre_counts >= threshold]
other_count = genre_counts[genre_counts < threshold].sum()

if other_count > 0:
    genre_counts_adjusted['Other'] = other_count

# Define a custom color palette
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#99e6e6', '#ffdb4d']
# if needed, you can add more colors to the list.

# Plot the pie chart
genre_counts_adjusted.plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    colors=colors,  # use the custom color palette
    wedgeprops=dict(width=0.4)  # if you want to make a donut chart
)

# Improve readability
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.ylabel('')  # Hide the y-axis label
plt.title('Genre Distribution')

# Show the plot
plt.show()

df_exploded = pieChartDF.explode('genres')

df_exploded['year_group'] = (df_exploded['release_year'] // 2) * 2

# 对year_group和genres进行分组并计数
genre_counts_over_time = df_exploded.groupby(['year_group', 'genres']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
genre_counts_over_time.plot(kind='line', ax=plt.gca())  # 使用 kind='line' 绘制折线图
plt.title('Genre Distribution Over Time')
plt.xlabel('Decade')
plt.ylabel('Number of Movies')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # 将图例移到图的外部
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

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


# 词频图----------------------------------------------------------------------------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



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
