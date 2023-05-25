from sklearn.neighbors import NearestNeighbors
from Controllers.MoviesPreprocessor import MoviePreprocessor
from Controllers.FeatureExtractor import FeatureExtractor
import os
from dotenv import load_dotenv

load_dotenv()

# Get the path to the Universal Sentence Encoder model from the environment variables
universal_sentence_encoder_model_path = os.environ.get("universal_sentence_encoder_model_path")


class MoviesRecommendationSystem:
    def __init__(self, n, user, movies, lang='en'):
        """
            Initialize the MoviesRecommendationSystem class.

             Args:
                 n (int): Number of recommended movies to retrieve.
                 user (dict): User information and preferences.
                 movies (list): List of movie data.
        """
        self.n = n
        self.user = user
        self.movies = MoviePreprocessor.preprocess_movies(movies, lang=lang)
        self.feature_extractor = FeatureExtractor(
            universal_sentence_encoder_model_path=universal_sentence_encoder_model_path)

        # Initialize the NearestNeighbors model with cosine distance metric and brute-force algorithm
        self.nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n)

    def __extract_features(self):
        """
            Extract features from movie titles using the Universal Sentence Encoder.

            Returns:
               array: Extracted features for each movie.
        """
        titles = [f'{movie["categories"]} {movie["description"]}' for movie in self.movies]
        return self.feature_extractor.fit_transform(titles)

    def __fit_nearest_neighbors(self):
        """
            Fit the NearestNeighbors model with the extracted movie features.
        """
        movies_features = self.__extract_features()
        self.nearest_neighbors.fit(movies_features)

    def __prepare_user_categories(self) -> str:
        """
           Prepare user categories for feature extraction.

           Returns:
               str: User categories as a single string.
        """
        user_categories = ""
        for category in self.user['categories']:
            user_categories += f'{category["name"]} '
        return user_categories

    def __get_user_features(self):
        """
          Extract features from user categories using the Universal Sentence Encoder.

          Returns:
              array: Extracted features for the user.
        """
        user_categories = self.__prepare_user_categories()
        return self.feature_extractor.transform([user_categories])

    def __get_neighbor_movies_ids(self, neighbors):
        """
           Retrieve the movie IDs of the nearest neighbor movies.

           Args:
               neighbors (array): Indices of the nearest neighbors.

           Returns:
               list: Movie IDs of the nearest neighbor movies.
        """
        movies_ids = []
        for n_index in neighbors:
            movies_ids.append(self.movies[n_index - 1]['id'])

        return movies_ids

    def get_recommended_movies(self):
        """
           Get recommended movies based on user preferences.

           Returns:
               list: Movie IDs of the recommended movies.
        """
        self.__fit_nearest_neighbors()

        user_features = self.__get_user_features()
        neighbors = self.nearest_neighbors.kneighbors(user_features, n_neighbors=self.n, return_distance=False)[0]

        return self.__get_neighbor_movies_ids(neighbors)
