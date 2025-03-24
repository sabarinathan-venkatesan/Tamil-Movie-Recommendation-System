#without poster of that movie

# # Step 1: Import Libraries
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Step 2: Load the Dataset
# @st.cache_data  # Cache the data to improve performance
# def load_data():
#     # Load the Tamil movies dataset
#     movies = pd.read_csv('tamil_movies.csv')
#     return movies

# movies = load_data()

# # Step 3: Preprocess the Data
# # Combine 'genres' and 'tags' into a single feature
# movies['combined_features'] = movies['Genres'] + " " + movies['Tags']

# # Step 4: Build the Recommendation System
# # Use TF-IDF to convert text into numerical vectors
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# # Step 5: Map Emotions to Keywords
# emotion_keywords = {
#     "happy": "feel-good comedy friendship",
#     "sad": "emotional drama heartwarming",
#     "romantic": "romance love emotional",
#     "angry": "intense dark crime",
#     "inspiring": "motivational life lessons social justice"
# }

# # Step 6: Create a Function to Get Recommendations
# def get_recommendations(emotion):
#     # Get keywords for the selected emotion
#     keywords = emotion_keywords.get(emotion, "")
#     if not keywords:
#         return pd.Series([])  # Return empty series if emotion not found
    
#     # Transform the keywords into a TF-IDF vector
#     keyword_vector = tfidf.transform([keywords])
    
#     # Compute cosine similarity between the keyword vector and all movies
#     cosine_sim = cosine_similarity(keyword_vector, tfidf_matrix)
    
#     # Get the top 5 most similar movies
#     sim_scores = list(enumerate(cosine_sim[0]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[:5]  # Get top 5 recommendations
#     movie_indices = [i[0] for i in sim_scores]
#     return movies['Title'].iloc[movie_indices]

# # Step 7: Create the Streamlit App
# st.title("Tamil Movie Recommendation System Based on Emotions")
# st.write("Enter your current emotion to get movie recommendations:")

# # Add a dropdown for emotions
# emotion = st.selectbox("Select your emotion:", ["happy", "sad", "romantic", "angry", "inspiring"])

# # Add a button to trigger recommendations
# if st.button("Get Recommendations"):
#     recommendations = get_recommendations(emotion)
#     if not recommendations.empty:
#         st.write(f"Top 5 {emotion.capitalize()} Tamil Movies:")
#         st.write(recommendations)
#     else:
#         st.write("No recommendations found for this emotion.")





# CUSTOM PROMPTS

# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the dataset
# @st.cache_data
# def load_data():
#     movies = pd.read_csv('tamil_movies.csv')
#     return movies

# movies = load_data()

# # Combine 'Genres' and 'Tags' into a single feature
# movies['combined_features'] = movies['Genres'] + " " + movies['Tags']

# # Use TF-IDF to convert text into numerical vectors
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# # Function to get recommendations based on user input
# def get_recommendations(user_input, tfidf_matrix, tfidf, movies):
#     # Transform user input into a TF-IDF vector
#     user_input_vector = tfidf.transform([user_input])
    
#     # Compute cosine similarity between user input and all movies
#     cosine_sim = cosine_similarity(user_input_vector, tfidf_matrix)
    
#     # Get the top 5 most similar movies
#     sim_scores = list(enumerate(cosine_sim[0]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[:5]  # Get top 5 recommendations
#     movie_indices = [i[0] for i in sim_scores]
#     return movies['Title'].iloc[movie_indices]

# # Streamlit app
# st.title("Tamil Movie Recommendation System")
# st.write("Enter your mood or preferences (e.g., 'I want a feel-good movie about friendship'):")

# # Add a text input for user prompt
# user_input = st.text_input("Write your prompt here:")

# # Add a button to trigger recommendations
# if st.button("Get Recommendations"):
#     if user_input:
#         recommendations = get_recommendations(user_input, tfidf_matrix, tfidf, movies)
#         st.write("Top 5 Recommendations:")
#         st.write(recommendations)
#     else:
#         st.write("Please enter a prompt to get recommendations.")



#With poster of that movie
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    movies = pd.read_csv('tamil_movies.csv')
    return movies

movies = load_data()

# Combine 'Genres' and 'Tags' into a single feature
movies['combined_features'] = movies['Genres'] + " " + movies['Tags']

# Use TF-IDF to convert text into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Function to get recommendations based on user input
def get_recommendations(user_input, tfidf_matrix, tfidf, movies):
    # Transform user input into a TF-IDF vector
    user_input_vector = tfidf.transform([user_input])
    
    # Compute cosine similarity between user input and all movies
    cosine_sim = cosine_similarity(user_input_vector, tfidf_matrix)
    
    # Get the top 6 most similar movies (to ensure even pairs)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:6]  # Get top 6 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Streamlit app
st.title("Tamil Movie Recommendation System")
st.write("Enter your mood or preferences (e.g., 'I want a feel-good movie about friendship'):")

# Add a text input for user prompt
user_input = st.text_input("Write your prompt here:")

# Add a button to trigger recommendations
if st.button("Get Recommendations"):
    if user_input:
        recommendations = get_recommendations(user_input, tfidf_matrix, tfidf, movies)
        st.write("Top Recommendations:")
        
        # Display two movie posters per row
        for i in range(0, len(recommendations), 2):  # Step by 2 for two posters per row
            cols = st.columns(2)  # Create two columns
            for j in range(2):  # Display two movies in the current row
                if i + j < len(recommendations):  # Check if there are enough movies left
                    row = recommendations.iloc[i + j]
                    with cols[j]:  # Use the current column
                        st.subheader(row['Title'])
                        
                        # Display movie poster
                        if pd.notna(row['Poster URL']):  # Check if Poster URL is not empty
                            st.image(row['Poster URL'], width=200)  # Display poster image
                        else:
                            st.write("Poster not available.")
                        
                        st.write(f"**Genres:** {row['Genres']}")
                        st.write(f"**Tags:** {row['Tags']}")
                        st.write("---")  # Add a separator
    else:
        st.write("Please enter a prompt to get recommendations.")