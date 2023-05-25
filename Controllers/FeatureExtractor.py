import tensorflow as tf
import tensorflow_hub


class FeatureExtractorType:
    def fit_transform(self, texts: [str]):
        pass

    def transform(self, texts: [str]):
        pass


class FeatureExtractor(FeatureExtractorType):
    def __init__(self, universal_sentence_encoder_model_path=None):
        """
            Initializes the FeatureExtractor with the path to the Universal Sentence Encoder model.

            Args:
              universal_sentence_encoder_model_path (str): Folder path of the trained model downloaded from TensorFlow Hub.
                  If None, the default model from TensorFlow Hub will be used.
        """

        self.universal_sentence_encoder_model_path = universal_sentence_encoder_model_path
        self.model = self.__load()

    def __load(self):
        """
            Loads the Universal Sentence Encoder model.

            Returns:
                tf.keras.Model: Loaded Universal Sentence Encoder model.
        """
        if self.universal_sentence_encoder_model_path is None:
            return tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return tf.keras.models.load_model(self.universal_sentence_encoder_model_path)

    def transform(self, texts: [str]):
        """
            Transforms the input texts into feature vectors.

            Args:
                texts (list): List of input texts.

            Returns:
                ndarray: Transformed feature vectors.
        """
        return self.model(texts)

    def fit_transform(self, texts: [str]):
        """
            Fits and transforms the input texts into feature vectors.

            Args:
                texts (list): List of input texts.

            Returns:
                ndarray: Transformed feature vectors.
        """
        return self.model(texts)
