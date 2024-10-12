import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
new_df = pd.read_csv('export2.csv')

# Vectorization: converting text data into a matrix of token counts
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()  # Transform the 'tags' column into vectors

# Uncomment below to see the first 10 vectors
# count = 0 
# for i in vector:
#     print(count, i)
#     count += 1
#     if count == 10:
#         break

# Uncomment below to see the first 10 feature names from the CountVectorizer
# count = 0 
# for i in cv.get_feature_names_out():
#     print(count, i)
#     count += 1
#     if count == 10:
#         break

# Calculate cosine similarity matrix from the vectors
similarity = cosine_similarity(vector)

def recommend(movie):
    """
    Recommends movies based on cosine similarity of tags.
    
    Parameters:
    movie (str): The title of the movie to base recommendations on.
    """
    # Normalize the input movie title for comparison
    movie = movie.title()  
    # Get the index of the movie in the DataFrame
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Get the similarity scores for that movie
    distances = similarity[movie_index]
    
    # Get the indices of the top 5 most similar movies (excluding the movie itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # Print the titles of the recommended movies
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Input from the user
name = input("Enter movie name: \a")
try:
    recommend(name)
except IndexError:
    print("Movie not found in database")
# Uncomment below to test with an empty input
# recommend('')  # This line can be used to test the function with an empty string.
