import re
import nltk
from nltk.stem import PorterStemmer
from tashaphyne.stemming import ArabicLightStemmer


class TextPreprocessor:
    @staticmethod
    def remove_non_alphabetic_characters(text):
        """
        Removes non-alphabetic characters from the text.

        Returns:
            str: The text with non-alphabetic characters removed.
        """
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def tokenize(text) -> [str]:
        """
           Tokenizes the text into individual words.

           Returns:
               list: A list of tokens (words).
        """
        return nltk.word_tokenize(text)

    @staticmethod
    def remove_stop_words_from_tokes(tokens: [str]) -> [str]:
        """
            Removes stop words from the list of tokens.

            Returns:
                list: A list of tokens with stop words removed.
        """
        stop_words = set(nltk.corpus.stopwords.words('arabic'))
        return [t for t in tokens if t not in stop_words]

    @staticmethod
    def stem(tokens: [str], lang='en') -> [str]:
        """
           Stems the tokens using either the PorterStemmer (for English) or the ArabicLightStemmer (for Arabic).

           Args:
               tokens (list): A list of tokens (words).
               lang (str, optional): Language of the text. Defaults to 'en'.

           Returns:
               list: A list of stemmed tokens.
        """
        if lang == 'ar':
            stemmer = ArabicLightStemmer().light_stem
        else:
            stemmer = PorterStemmer().stem

        return [stemmer(token) for token in tokens]

    @staticmethod
    def combine_words_list(words: [str]) -> str:
        """
            Combines a list of words into a single string.
        """
        return ' '.join(words)

    @staticmethod
    def preprocess(text, lang) -> str:
        """
          Preprocesses the text by removing non-alphabetic characters, tokenizing, removing stop words, and stemming.

          Args:
              text (str): The input text.
              lang (str): Language of the text.

          Returns:
              str: The preprocessed text.
        """
        text_with_no_alphabet = TextPreprocessor.remove_non_alphabetic_characters(text)

        tokens = TextPreprocessor.tokenize(text_with_no_alphabet)
        tokens_with_no_stop_words = TextPreprocessor.remove_stop_words_from_tokes(tokens)

        stemmed_tokens = TextPreprocessor.stem(tokens_with_no_stop_words, lang=lang)

        return TextPreprocessor.combine_words_list(stemmed_tokens)
