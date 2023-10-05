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
from sklearn.model_selection import cross_val_score

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
numeric_transformer = SimpleImputer(strategy='median')  # Impute missing values using the median

# Define categorical features and the corresponding transformer
categorical_features = ['genres', 'production_countries']
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
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
mse1 = mean_squared_error(Y_test, predictions)
print(f'Mean Squared Error: {mse1}')

# By examining MSE, this model is not ideal, which may be because people's taste for movies
# has changed with the change of times


# preprocessor and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Performing cross-validation
scores = cross_val_score(pipeline, X, Y, cv=10, scoring='neg_mean_squared_error')

# Cross-validation returns negative MSE values because the scoring function
mse_scores = -scores
mse2 = np.mean(mse_scores)
print(f'Mean Squared Error: {mse2}')


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
minimum = 0.02 * total

# Group genres with counts below the threshold into an "Other" category
genre_counts_adjusted = genre_counts[genre_counts >= minimum]
others = genre_counts[genre_counts < minimum].sum()

if others > 0:
    genre_counts_adjusted['Other'] = others

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
plt.xlabel('Decade')
plt.ylabel('Number of Movies')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('PNG/Genre Distribution Over Time.png', bbox_inches='tight')

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
wordcount = 20
topwords = word_count.most_common(wordcount)

# Reverse the list of common words to display in descending order
topwords.reverse()

# Plot the most common words as a horizontal bar chart
plt.figure(figsize=(10, 5))
plt.barh([word[0] for word in topwords], [word[1] for word in topwords], color='skyblue')
plt.xlabel('Count')
plt.title(f'Top {wordcount} Common Words in Movie Descriptions')

# Annotate the bars with the word count
for i, count in enumerate([word[1] for word in topwords]):
    plt.text(count + 10, i, str(count), va='center', fontsize=10, color='black')

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
df_of_movie = titles[titles['type'] == 'MOVIE']

# Create dictionaries to store the sum of votes and number of occurrences for each genre
dictionary_genre_sumvotes = {}
dictionary_genrecount = {}

# Iterate through each row in the filtered movies dataframe
for index, row in df_of_movie.iterrows():
    genres = row['genres']  # Extract the genres for the current movie
    imdb_votes = row['imdb_votes']  # Extract the IMDB votes for the current movie

    # Iterate through each genre of the current movie
    for genre in genres:
        # Update the votes sum for the current genre
        if genre in dictionary_genre_sumvotes:
            dictionary_genre_sumvotes[genre] += imdb_votes
        else:
            dictionary_genre_sumvotes[genre] = imdb_votes

        # Update the count of the current genre
        if genre in dictionary_genrecount:
            dictionary_genrecount[genre] += 1
        else:
            dictionary_genrecount[genre] = 1

# Calculate the average votes for each genre
dictionary_genre_avgvotes = {genre: dictionary_genre_sumvotes[genre] / dictionary_genrecount[genre] for genre in dictionary_genre_sumvotes}

# Print out the average votes for each genre
for genre, avgvotes in dictionary_genre_avgvotes.items():
    print(f"{genre}: {avgvotes:.2f}")

# Get the list of genres and their corresponding average votes for plotting
genres = list(dictionary_genre_avgvotes.keys())
avgvotes = list(dictionary_genre_avgvotes.values())

# Plotting
plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, len(genres)))
bars = plt.barh(genres, avgvotes, color=colors)

# Annotate each bar with the average votes value
for bar, votes in zip(bars, avgvotes):
    plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2, f'{votes:.0f}',
             va='center', ha='right', color='black', fontsize=10)

# Set labels, title, and grid for the plot
plt.xlabel('Average IMDB Votes')
plt.ylabel('Genre')
plt.title('Average IMDB Votes by Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlim([0, max(avgvotes) + 0.5])
plt.savefig('PNG/Average IMDB Votes by Genre', bbox_inches='tight')
plt.show()  # Display the plot

# top 10 director-------------------------------------------------------------------------------------------------------

# Load the 'credits' dataset
df2 = pd.read_csv("credits.csv", encoding='ISO-8859-1')

# Filter the dataset to only include rows where the role is 'DIRECTOR'
df2 = df2[df2["role"] == "DIRECTOR"]

# Merge the 'titles' and 'df2' datasets on the 'id' column
data_director = pd.merge(titles, df2, on='id')

# Group the merged data by director's name and calculate the mean 'imdb_score' for each director
director_of_avgscore = data_director.groupby('name')['imdb_score'].mean()

# Sort the directors by their average 'imdb_score' in descending order and get the top 10
top10director = director_of_avgscore.sort_values(ascending=False).head(10)

# Define a list of colors for the bars in the plot
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#99e6e6', '#ffdb4d', '#d9b3ff', '#ff6666']

# Create a new figure for plotting
plt.figure(figsize=(10, 6))

# Plot the average IMDb scores for the top 10 directors as a horizontal bar chart
bars = top10director.plot(kind='barh', color=colors)

# Set the title and axis labels for the plot
plt.title('Top 10 Directors by Average TMDB Score')
plt.xlabel('Average IMDB Score')
plt.ylabel('Director')

# Annotate each bar in the plot with the exact IMDb score
for index, value in enumerate(top10director):
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
titles_of_movie = titles[titles['type'] == 'MOVIE'].copy()


# Creates a new column to store the category of the movie
def classification_runtime(runtime):
    if runtime < 40:
        return 'Short movie'
    elif 40 <= runtime <= 70:
        return 'Medium-length movie'
    else:
        return 'Feature movie'


titles_of_movie['category'] = titles_of_movie['runtime'].apply(classification_runtime)

# Count the number of movies in each category
category_counts = titles_of_movie['category'].value_counts()

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
titles_of_movie = titles[titles['type'] == 'MOVIE'].copy()

# Filters out the specified five age_certification categories
certification = ['G', 'NC-17', 'PG', 'PG-13', 'R']
titles_of_movie = titles_of_movie[titles_of_movie['age_certification'].isin(certification)]

# Count the number of movies in each category
counts_of_classification = titles_of_movie['age_certification'].value_counts()

ax = counts_of_classification.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'purple'])
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
counts_of_imdb = merged_data.groupby('imdb_score').size()

# Create a line plot of IMDb score distribution
plt.figure()
counts_of_imdb.plot()

# Add title and axis labels
plt.title('IMDb Score Distribution')
plt.xlabel('IMDb Score')
plt.ylabel('Frequency')

# Show the plot
plt.savefig('PNG/IMDb Score Distribution', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Check out the top 10 actor with the highest average imdb rating

data_of_actor = merged_data[merged_data['role'] == 'ACTOR']
# Calculate the average imdb score for each character
actor_avgscore = data_of_actor.groupby('name')['imdb_score'].mean()

# Sort and get the top 10 characters with the highest average rating
top10actor = actor_avgscore.sort_values(ascending=False).head(10)

print(top10actor)

color_list = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan']

plt.figure(figsize=(10, 6))
top10actor.plot(kind='barh', color=color_list)

plt.title('Top 10 actor by Average IMDb Score')
plt.xlabel('Average IMDb Score')
plt.ylabel('Actor')

for index, value in enumerate(top10actor):
    plt.text(value, index, f'{value:.2f}')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('PNG/Top 10 Actor by Average IMDb Score', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Average imdb_score for each movie category

df = pd.read_csv('titles.csv')

df_of_movie = df[df['type'] == 'MOVIE']

dictionarygenrescore_sum = {}
dictionarygenrecount = {}

for index, row in df_of_movie.iterrows():
    genres = eval(row['genres'])
    imdb_score = row['imdb_score']
    for genre in genres:

        if genre in dictionarygenrescore_sum:
            dictionarygenrescore_sum[genre] += imdb_score
        else:
            dictionarygenrescore_sum[genre] = imdb_score

        if genre in dictionarygenrecount:
            dictionarygenrecount[genre] += 1
        else:
            dictionarygenrecount[genre] = 1

# Average the ratings for each type
dictionary_genre_avgscore = {genre: dictionarygenrescore_sum[genre] / dictionarygenrecount[genre] for genre in dictionarygenrescore_sum}

for genre, avg_score in dictionary_genre_avgscore.items():
    print(f"{genre}: {avg_score:.2f}")

genres = list(dictionary_genre_avgscore.keys())
avgscores = list(dictionary_genre_avgscore.values())

plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, len(genres)))
bars = plt.barh(genres, avgscores, color=colors)

for bar, score in zip(bars, avgscores):
    plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2, f'{score:.2f}',
             va='center', ha='right', color='black', fontsize=10)

plt.xlabel('Average IMDB Score')
plt.ylabel('Genre')
plt.title('Average IMDB Score by Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlim([0, max(avgscores) + 0.5])
plt.savefig('PNG/Average IMDB Score by Genre', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------------------------------
