from dotenv import load_dotenv
import os
import pandas as pd
from Controllers.Text import TextPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# Paths to the datasets for Arabic and English spam detection
arabic_spam_detector_data_set_path = os.environ.get("arabic_spam_detector_data_set_path")
english_spam_detector_data_set_path = os.environ.get("english_spam_detector_data_set_path")


class SpamDetector:
    def __init__(self, content_column_name: str, type_column_name: str, lang='en'):
        """
            Initializes the SpamDetector class.

           Args:
               content_column_name (str): Name of the column containing the content/text.
               type_column_name (str): Name of the column containing the type/label.
               lang (str): Language of the dataset. Defaults to 'en' (English).
        """
        self.content_column_name = content_column_name
        self.type_column_name = type_column_name
        self.lang = lang

        # Choose the dataset path based on the language
        dataset_path = arabic_spam_detector_data_set_path if lang == 'ar' else english_spam_detector_data_set_path

        # Read the dataset into a Pandas dataframe
        self.dataframe = pd.read_csv(dataset_path, delimiter=',', names=[content_column_name, type_column_name])
        self.model = MultinomialNB()
        self.vectorizer = TfidfVectorizer()

    def apply_preprocessing(self, column: str):
        """
            Applies preprocessing to the specified column of the dataframe.

            Args:
                column (str): Name of the column to apply preprocessing to.
        """
        self.dataframe[column] = self.dataframe[column].apply(TextPreprocessor.preprocess, lang=self.lang)

    def fit_model(self):
        """
            Fits the spam detection model using the preprocessed data.
        """
        self.apply_preprocessing(self.content_column_name)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe[self.content_column_name],
                                                            self.dataframe[self.type_column_name],
                                                            test_size=0.2)

        # Vectorize the training data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)

        # Train the model using the vectorized data
        self.model.fit(X_train_vectorized, y_train)

    def is_spam(self, text):
        """
          Determines if the given text is spam or not.

          Args:
              text (str): The text to classify.

          Returns:
              bool: True if the text is classified as spam, False otherwise.
        """
        preprocessed_comment = TextPreprocessor.preprocess(text, lang=self.lang)
        vectorized_comment = self.vectorizer.transform([preprocessed_comment])
        prediction = self.model.predict(vectorized_comment)
        return prediction[0] == '1'

