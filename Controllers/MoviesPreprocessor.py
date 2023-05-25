from Controllers.Text import TextPreprocessor


class MoviePreprocessor:
    @staticmethod
    def preprocess_movies(movies, lang):
        """
           Preprocesses the movie data by applying text preprocessing to descriptions and categories.

           Args:
               movies (list): List of movie dictionaries.
               lang (str): the language of the movies description 
           Returns:
               list: Preprocessed movie data.
        """
        for index in range(len(movies) - 1):
            MoviePreprocessor.preprocess_description(index, movies, lang)
            MoviePreprocessor.preprocess_categories(index, movies)

        return movies

    @staticmethod
    def preprocess_description(index, movies, lang):
        """
          Preprocesses the description of a movie by applying text preprocessing.

          Args:
              index (int): Index of the movie in the list.
              movies (list): List of movie dictionaries.
              lang
        """
        movies[index]['description'] = TextPreprocessor.preprocess(movies[index]['description'], lang=lang)

    @staticmethod
    def preprocess_categories(index, movies):
        """
           Preprocesses the categories of a movie by combining category names into a single string.

           Args:
               index (int): Index of the movie in the list.
               movies (list): List of movie dictionaries.
        """
        categories = ""
        for category in movies[index]['categories']:
            categories += f'{category["name"]} '

        movies[index]["categories"] = categories
