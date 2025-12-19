import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data():
    df = pd.read_csv("dataset/movies.csv")
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['overview'] + " " + df['genres']
    return df

# Create similarity matrix
def create_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

# Recommend movies
def recommend(movie_name, df, similarity):
    if movie_name not in df['title'].values:
        return ["Movie not found in dataset"]

    index = df[df['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(df.iloc[i[0]].title)

    return recommended_movies
