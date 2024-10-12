import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

new_df = pd.read_csv('export2.csv')
#vectorisation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

count = 0 
# print(vector)
# for i in vector:
    
#     print(count,i)
#     count = count+1
#     if count==10:
#         break
# print(cv.get_feature_names_out())
count = 0 

# for i in cv.get_feature_names_out():
    
#     print(count,i)
#     count = count+1
#     if count==10:
#         break


similarity = cosine_similarity(vector)
# print(similarity.shape)
# print(similarity[0])
# print(similarity[1])

def recommend(movie):
    movie = movie.title()
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x : x[1])[0:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# print(new_df[new_df['title'] == "Batman Begins"]  )
# print(new_df[new_df['title'] == "Batman Begins"].index[0]  )
# print(new_df[new_df['title'] == "Batman Begins"].index[1]  ) #error
name = input("Enter movie name : \a")
try:
    recommend(name)
except:
    print("Movie not found in database")
# recommend('')