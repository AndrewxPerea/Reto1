from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cargar y preparar datos (simplificado; idealmente cargar al inicio)
df_tracks = pd.read_csv('tracks.csv', encoding='utf-8')
features_content = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]
df_tracks = df_tracks.dropna(subset=features_content + ['popularity', 'release_date'])
df_tracks['artists'] = df_tracks['artists'].str.lower().str.strip()
df_tracks['explicit'] = df_tracks['explicit'].astype(bool)
scaler = StandardScaler()
X_content = scaler.fit_transform(df_tracks[features_content])

# Simulación simple matriz usuario-canción para colaborativo
user_song_matrix = np.zeros((5, df_tracks.shape[0]))
user_song_matrix[0, [10, 100, 500]] = 1
user_song_matrix[1, [10, 50, 200]] = 1
user_song_matrix[2, [100, 300, 400]] = 1
user_song_matrix[3, [500, 20, 30]] = 1
user_song_matrix[4, [7, 50, 70]] = 1

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        # Leer índices de canciones favoritas ingresados por el usuario
        fav_indices_str = request.form.get('favorites')
        try:
            user_favorite_indices = list(map(int, fav_indices_str.split(',')))
        except:
            user_favorite_indices = [10, 100, 500]  # Default

        # Filtrado contenido
        user_fav_vectors = X_content[user_favorite_indices]
        user_profile = np.mean(user_fav_vectors, axis=0).reshape(1, -1)
        sim_content = cosine_similarity(user_profile, X_content).flatten()

        # Filtrado colaborativo
        target_user = 0
        similar_users = cosine_similarity(user_song_matrix)[target_user]
        score_collab = np.dot(similar_users, user_song_matrix) / (similar_users.sum() + 1e-6)

        def normalize(scores):
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)

        score_content_norm = normalize(sim_content)
        score_collab_norm = normalize(score_collab)

        alpha = 0.6
        score_hybrid = alpha * score_content_norm + (1 - alpha) * score_collab_norm

        ranking_indices = np.argsort(score_hybrid)[::-1]
        known_indices = set(user_favorite_indices)
        final_recommendations = [idx for idx in ranking_indices if idx not in known_indices]
        top_n = 10
        top_recommendations = final_recommendations[:top_n]

        recommendations = df_tracks.iloc[top_recommendations][['name', 'artists', 'popularity']].to_dict(orient='records')

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
