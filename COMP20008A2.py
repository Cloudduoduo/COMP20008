import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
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
titles['genres'] = titles['genres'].apply(ast.literal_eval)

# print(credit.info())
# print(titles.info())


# ------------------------------------------------------------------------------------------------------------


features = ['release_year', 'runtime', 'tmdb_popularity', 'tmdb_score', 'genres', 'production_countries']
target = 'imdb_score'

X = titles[features]
Y = titles[target]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=666)

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

print(titles.dtypes)

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

# Check out the top 10 characters with the highest average imdb rating

# Calculate the average imdb score for each character
character_avg_score = merged_data.groupby('character')['imdb_score'].mean()

# Sort and get the top 10 characters with the highest average rating
top10_characters = character_avg_score.sort_values(ascending=False).head(10)

print(top10_characters)

color_list = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan']

plt.figure(figsize=(10, 6))
top10_characters.plot(kind='barh', color=color_list)

plt.title('Top 10 Characters by Average IMDb Score')
plt.xlabel('Average IMDb Score')
plt.ylabel('Character')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
