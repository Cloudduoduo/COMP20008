import numpy as np
import ast
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud

# pre-processing--------------------------------------------------------------------------------------------------------

# Load the datasets
credit = pd.read_csv('credits.csv', encoding='ISO-8859-1')
titles = pd.read_csv('titles.csv', encoding='ISO-8859-1')

# Remove rows with missing values from the 'credit' dataframe
credit.dropna(inplace=True)

# Remove rows with missing values in specified columns from the 'titles' dataframe
columns_to_check = ['imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
titles.dropna(subset=columns_to_check, inplace=True)

# Save the cleaned 'titles' dataframe back to a CSV file
titles.to_csv('titles.csv', index=False)

# Remove duplicate rows from both dataframes
credit.drop_duplicates(inplace=True)
titles.drop_duplicates(inplace=True)

# change variable type
titles['genres'] = titles['genres'].apply(ast.literal_eval)
titles['production_countries'] = titles['production_countries'].apply(ast.literal_eval)

# Check dataframe
# print(credit.info())
# print(titles.info())

# Machine learning------------------------------------------------------------------------------------------------------


# Define the features and the target variable for our model
features = ['release_year', 'runtime', 'tmdb_popularity', 'tmdb_score', 'genres', 'production_countries']
target = 'imdb_score'

# Split the dataset into features (X) and target (Y)
X = titles[features]
Y = titles[target]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=666)

# Define numeric features and the corresponding transformer
numeric_features = ['release_year', 'runtime', 'tmdb_popularity', 'tmdb_score']
numeric_transformer = SimpleImputer(strategy='median')

# Define categorical features and the corresponding transformer
categorical_features = ['genres', 'production_countries']
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing values with 'missing' label
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Apply one-hot encoding to categorical features
])

# Create a column transformer that applies the above transformations to the respective columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the modeling pipeline: pre-processing followed by a linear regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model on the training data
model.fit(X_train, Y_train)

# Predict the target values for the test set
predictions = model.predict(X_test)

# Calculate and print the mean squared error of the predictions
mse = mean_squared_error(Y_test, predictions)
print(f'Mean Squared Error: {mse}')

# By examining MSE, this model is not ideal, which may be because people's taste for movies
# has changed with the change of times


# pie chart and line chart about genre and years------------------------------------------------------------------------

# Convert the genres column in titles dataframe to a list type
pieChartDF = titles
pieChartDF['genres'] = pieChartDF['genres'].apply(
    ast.literal_eval)  # Convert string representations of lists to actual lists

# Split each list in the genres column into separate rows
p_exploded = pieChartDF['genres'].explode()

# Calculate the frequency of each genre
genre_counts = p_exploded.value_counts()
total = genre_counts.sum()

# Set a threshold for minimum count required for a genre to be displayed separately in the pie chart
threshold = 0.02 * total

# Group genres with counts below the threshold into an "Other" category
genre_counts_adjusted = genre_counts[genre_counts >= threshold]
other_count = genre_counts[genre_counts < threshold].sum()

if other_count > 0:
    genre_counts_adjusted['Other'] = other_count

# Define custom colors for the pie chart
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#99e6e6', '#ffdb4d']

# Create and display the pie chart with adjusted genre counts
genre_counts_adjusted.plot.pie(
    autopct='%1.1f%%',  # Display percentages on the chart
    startangle=90,  # Starting angle for the first slice
    counterclock=False,  # Display slices in a counter-clockwise fashion
    colors=colors,  # Use the custom color palette
    wedgeprops=dict(width=0.4)  # Create a donut chart by defining a wedge width
)

plt.axis('equal')  # Ensure the pie chart is circular
plt.ylabel('')  # Remove the y-axis label
plt.title('Genre Distribution')  # Set the title for the pie chart
plt.savefig('PNG/Genre Distribution.png', bbox_inches='tight')
# Display the pie chart
plt.show()

# Split the genres in the pieChartDF into separate rows for further analysis
df_exploded = pieChartDF.explode('genres')

# Group the release years in 2-year bins
df_exploded['year_group'] = (df_exploded['release_year'] // 2) * 2

# Calculate the frequency of each genre over the binned years
genre_counts_over_time = df_exploded.groupby(['year_group', 'genres']).size().unstack(fill_value=0)

# Plot the genre distribution over time
plt.figure(figsize=(10, 6))
genre_counts_over_time.plot(kind='line', ax=plt.gca())
plt.title('Genre Distribution Over Time')
plt.xlabel('Decade')  # Set x-axis label
plt.ylabel('Number of Movies')  # Set y-axis label
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # Place the legend outside the plot
plt.savefig('PNG/Genre Distribution Over Time.png', bbox_inches='tight')
# Display the line chart
plt.show()

# Word frequency plot and word cloud plot-------------------------------------------------------------------------------

# Fill any missing descriptions with empty strings
titles['description'].fillna('', inplace=True)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up the list of English stop words
stop_words = set(stopwords.words('english'))

# Initialize a counter for word frequencies
word_count = Counter()

# Loop through each movie description
for desc in titles['description']:
    # Convert to lowercase and remove punctuation
    desc = desc.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenize the description into words
    words = nltk.word_tokenize(desc)
    # Update the word count, excluding stop words
    word_count.update(word for word in words if word not in stop_words)

# Get the most common words
top_n = 20
common_words = word_count.most_common(top_n)

# Plot the most common words as a horizontal bar chart
plt.figure(figsize=(10, 5))
plt.barh([word[0] for word in common_words], [word[1] for word in common_words], color='skyblue')
plt.xlabel('Count')
plt.title(f'Top {top_n} Common Words in Movie Descriptions')
plt.savefig('PNG/Common Words in Movie Descriptions.png', bbox_inches='tight')
plt.show()

# Generate a word cloud based on the word frequencies
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      max_words=200, colormap='viridis').generate_from_frequencies(word_count)

# Display the word cloud
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')  # Hide axes for better visual
plt.savefig('PNG/Word Cloud.png', bbox_inches='tight')
plt.show()

# Average imdb_votes for each movie category----------------------------------------------------------------------------


# Filter the titles dataframe to only get entries where the type is 'MOVIE'
movie_df = titles[titles['type'] == 'MOVIE']

# Create dictionaries to store the sum of votes and number of occurrences for each genre
genre_votes_sum_dict = {}
genre_count_dict = {}

# Iterate through each row in the filtered movies dataframe
for index, row in movie_df.iterrows():
    genres = row['genres']  # Extract the genres for the current movie
    imdb_votes = row['imdb_votes']  # Extract the IMDB votes for the current movie

    # Iterate through each genre of the current movie
    for genre in genres:
        # Update the votes sum for the current genre
        if genre in genre_votes_sum_dict:
            genre_votes_sum_dict[genre] += imdb_votes
        else:
            genre_votes_sum_dict[genre] = imdb_votes

        # Update the count of the current genre
        if genre in genre_count_dict:
            genre_count_dict[genre] += 1
        else:
            genre_count_dict[genre] = 1

# Calculate the average votes for each genre
genre_avg_votes_dict = {genre: genre_votes_sum_dict[genre] / genre_count_dict[genre] for genre in genre_votes_sum_dict}

# Print out the average votes for each genre
for genre, avg_votes in genre_avg_votes_dict.items():
    print(f"{genre}: {avg_votes:.2f}")

# Get the list of genres and their corresponding average votes for plotting
genres = list(genre_avg_votes_dict.keys())
avg_votes = list(genre_avg_votes_dict.values())

# Plotting
plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, len(genres)))
bars = plt.barh(genres, avg_votes, color=colors)

# Annotate each bar with the average votes value
for bar, votes in zip(bars, avg_votes):
    plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2, f'{votes:.0f}',
             va='center', ha='right', color='black', fontsize=10)

# Set labels, title, and grid for the plot
plt.xlabel('Average IMDB Votes')
plt.ylabel('Genre')
plt.title('Average IMDB Votes by Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlim([0, max(avg_votes) + 0.5])
plt.savefig('PNG/Average IMDB Votes by Genre', bbox_inches='tight')
plt.show()  # Display the plot

# top 10 director-------------------------------------------------------------------------------------------------------

# Load the 'credits' dataset
df2 = pd.read_csv("credits.csv", encoding='ISO-8859-1')

# Filter the dataset to only include rows where the role is 'DIRECTOR'
df2 = df2[df2["role"] == "DIRECTOR"]

# Merge the 'titles' and 'df2' datasets on the 'id' column
director_data = pd.merge(titles, df2, on='id')

# Group the merged data by director's name and calculate the mean 'imdb_score' for each director
director_avg_score = director_data.groupby('name')['imdb_score'].mean()

# Sort the directors by their average 'imdb_score' in descending order and get the top 10
top10_director = director_avg_score.sort_values(ascending=False).head(10)

# Define a list of colors for the bars in the plot
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#99e6e6', '#ffdb4d', '#d9b3ff', '#ff6666']

# Create a new figure for plotting
plt.figure(figsize=(10, 6))

# Plot the average IMDb scores for the top 10 directors as a horizontal bar chart
bars = top10_director.plot(kind='barh', color=colors)

# Set the title and axis labels for the plot
plt.title('Top 10 Directors by Average TMDB Score')
plt.xlabel('Average IMDB Score')
plt.ylabel('Director')

# Annotate each bar in the plot with the exact IMDb score
for index, value in enumerate(top10_director):
    plt.text(value, index, f'{value:.2f}')

# Invert the order of the y-axis for better visualization
plt.gca().invert_yaxis()

# Adjust the layout for the plot
plt.tight_layout()
plt.savefig('PNG/Top 10 Directors by Average TMDB Score', bbox_inches='tight')
# Display the plot
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
plt.savefig('PNG/Distribution of Movie Lengths', bbox_inches='tight')
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
plt.savefig('PNG/Distribution of Age Certifications', bbox_inches='tight')
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
plt.savefig('PNG/IMDb Score Distribution', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Check out the top 10 actor with the highest average imdb rating

actor_data = merged_data[merged_data['role'] == 'ACTOR']
# Calculate the average imdb score for each character
actor_avg_score = actor_data.groupby('name')['imdb_score'].mean()

# Sort and get the top 10 characters with the highest average rating
top10_actor = actor_avg_score.sort_values(ascending=False).head(10)

print(top10_actor)

color_list = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan']

plt.figure(figsize=(10, 6))
top10_actor.plot(kind='barh', color=color_list)

plt.title('Top 10 actor by Average IMDb Score')
plt.xlabel('Average IMDb Score')
plt.ylabel('Actor')

for index, value in enumerate(top10_actor):
    plt.text(value, index, f'{value:.2f}')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('PNG/Top 10 Actor by Average IMDb Score', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Average imdb_score for each movie category

df = pd.read_csv('titles.csv')

movie_df = df[df['type'] == 'MOVIE']

genre_score_sum_dict = {}
genre_count_dict = {}

for index, row in movie_df.iterrows():
    genres = eval(row['genres'])
    imdb_score = row['imdb_score']
    for genre in genres:

        if genre in genre_score_sum_dict:
            genre_score_sum_dict[genre] += imdb_score
        else:
            genre_score_sum_dict[genre] = imdb_score

        if genre in genre_count_dict:
            genre_count_dict[genre] += 1
        else:
            genre_count_dict[genre] = 1

# Average the ratings for each type
genre_avg_score_dict = {genre: genre_score_sum_dict[genre] / genre_count_dict[genre] for genre in genre_score_sum_dict}

for genre, avg_score in genre_avg_score_dict.items():
    print(f"{genre}: {avg_score:.2f}")

genres = list(genre_avg_score_dict.keys())
avg_scores = list(genre_avg_score_dict.values())

plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, len(genres)))
bars = plt.barh(genres, avg_scores, color=colors)

for bar, score in zip(bars, avg_scores):
    plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2, f'{score:.2f}',
             va='center', ha='right', color='black', fontsize=10)

plt.xlabel('Average IMDB Score')
plt.ylabel('Genre')
plt.title('Average IMDB Score by Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlim([0, max(avg_scores) + 0.5])
plt.savefig('PNG/Average IMDB Score by Genre', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------
