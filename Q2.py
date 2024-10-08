import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# Load the dataset
df = pd.read_csv('imdb_dataset.csv')

# Select the relevant features
features = ['Movie Name', 'IMDB Rating', 'Genre', 'Cast', 'Director', 'Metascore']
df = df[features]

# Encode categorical features
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Cast'] = le.fit_transform(df['Cast'])
df['Director'] = le.fit_transform(df['Director'])

# Create a user-item interaction matrix
user_item_matrix = pd.pivot_table(df, values='IMDB Rating', index='Movie Name', columns='Genre')

# Normalize the ratings
user_item_matrix = user_item_matrix.apply(lambda x: (x - x.mean()) / x.std())

# Define the autoencoder architecture
input_dim = user_item_matrix.shape[1]
latent_dim = 10

input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(latent_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

autoencoder = Model(input_layer, decoder_layer)

# Define the encoder model
encoder = Model(input_layer, encoder_layer)

# Define the decoder model
decoder_input = Input(shape=(latent_dim,))
decoder_output = autoencoder.layers[-1](decoder_input)
decoder = Model(decoder_input, decoder_output)

# Compile the autoencoder
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

# Compile the encoder and decoder
encoder.compile(loss='binary_crossentropy', optimizer='adam')
decoder.compile(loss='binary_crossentropy', optimizer='adam')

# Train the autoencoder
autoencoder.fit(user_item_matrix, user_item_matrix, epochs=10, batch_size=32, verbose=2)


# Define a function to make recommendations
def make_recommendations(movie_name, num_recs):
    # Get the movie's latent representation
    movie_latent = encoder.predict(user_item_matrix.loc[movie_name].values.reshape(1, -1))

    # Reshape movie_latent to have shape (1981, 10)
    movie_latent = movie_latent.reshape(1981, 10)

    # Calculate the cosine similarity between the input movie and all other movies
    similarities = np.dot(user_item_matrix.values.T, movie_latent) / (
                np.linalg.norm(user_item_matrix.values, axis=1) * np.linalg.norm(movie_latent, axis=1))

    # Get the top-N recommended movies
    recommended_indices = np.argsort(-similarities)[:num_recs]

    # Map the indices back to the original movie names
    recommended_movies = user_item_matrix.index[recommended_indices]

    return recommended_movies


# Test the recommendation function
movie_name = '10 Things I Hate About You'
num_recs = 5
recommended_movies = make_recommendations(movie_name, num_recs)
print(recommended_movies)
