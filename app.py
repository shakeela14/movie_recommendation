import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# -------------------------------
# Load and preprocess dataset
# -------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['id', 'title', 'genres', 'keywords', 'overview']]

    # Convert JSON columns to string
    def convert(text):
        L = []
        try:
            for i in ast.literal_eval(text):
                L.append(i['name'])
        except:
            pass
        return " ".join(L)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['overview'] = movies['overview'].fillna("")
    movies['tags'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']
    return movies

movies = load_data()

# -------------------------------
# Build similarity matrix
# -------------------------------
@st.cache_resource
def build_model():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = build_model()

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# -------------------------------
# Custom CSS for Netflix-style UI
# -------------------------------
st.markdown("""
    <style>
        /* Full background black */
        .stApp {
            background-color: #000000;
        }

        /* Main title in Netflix red */
        .title, h1 {
            text-align: center;
            color: #e50914;
            font-size: 60px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Subheaders & description in white */
        .subtitle, .subheader, h2, h3, h4, p {
            text-align: center; 
            color: white !important;
            font-size: 20px;
            margin-bottom: 20px;
        }

        /* Selectbox label in white */
        div[data-baseweb="select"] label {
            color: white !important;
            font-weight: bold;
            font-size: 18px;
        }

        /* Dropdown menu style */
        div[data-baseweb="select"] > div[class*="menu"] {
            background-color: #1c1c1c !important;
            color: white !important;
        }
        div[data-baseweb="select"] > div[class*="menu"] div {
            color: white !important;
        }
        div[data-baseweb="select"] > div[class*="menu"] div:hover {
            background-color: #E50914 !important;
            color: white !important;
        }

        /* Recommendation cards */
        .movie-card {
            background-color: #141414;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            color: white;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            transition: transform 0.2s ease-in-out;
            cursor: pointer;
        }
        .movie-card:hover {
            transform: scale(1.08);
            background-color: #E50914;
        }

        /* Responsive grid for movie cards */
        .movie-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        /* Style the Recommend button */
        div.stButton > button {
            background-color: #e50914;  /* Netflix red background */
            color: white;                /* White text */
            font-size: 18px;
            font-weight: bold;
            padding: 10px 25px;
            border-radius: 8px;
            border: none;
            transition: transform 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            transform: scale(1.05);       /* Slight hover effect */
            background-color: #ff1a1a;    /* Slightly brighter red on hover */
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------------
# Streamlit UI Layout
# -------------------------------
# Main Header
st.markdown("<h1 class='title' style='color:red'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

# Subheading / Description
st.markdown("<h3 class='subtitle'>Discover your next favorite film instantly! 🍿</h3>", unsafe_allow_html=True)

# Engaging tagline
st.markdown("<p>⭐ Your personalized movie journey starts here — pick one and we’ll surprise you!</p>", unsafe_allow_html=True)

# Movie selection dropdown
selected_movie = st.selectbox("🎥 Choose a movie:", movies['title'].values)

# Recommend button
if st.button("✨ Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader("🍿 Top 5 Recommendations for you:")
        # Display cards in responsive grid
        movie_html = "<div class='movie-grid'>"
        for movie in recommendations:
            movie_html += f"<div class='movie-card'>🎬 {movie}</div>"
        movie_html += "</div>"
        st.markdown(movie_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Movie not found in dataset.")
