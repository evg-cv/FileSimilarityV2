import gensim
import numpy as np

from operator import add
from nltk.data import find
from utils.folder_file_manager import log_print
from settings import MODEL_PATH


class GFeatureExtractor:
    def __init__(self):
        word2vec_sample = str(find(MODEL_PATH))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

    @staticmethod
    def calculate_text_feature(word_features):
        text_feature = word_features[0]
        for w_feature in word_features[1:]:
            text_feature = list(map(add, text_feature, w_feature))

        return text_feature

    def get_feature_token_words(self, text):
        # sentences = self.text_processor.tokenize_sentence(text=text)
        text_features = []

        for t_word in text:
            if t_word.lower() in ['the', 'and', 'are', 'a']:
                continue
            try:
                word_feature = self.model[t_word.lower()]
                text_features.append(word_feature)
            except Exception as e:
                log_print(e)

        try:
            text_feature = self.calculate_text_feature(word_features=text_features)
        except Exception as e:
            log_print(e)
            text_feature = np.zeros(900)

        return text_feature


if __name__ == '__main__':
    GFeatureExtractor().get_feature_token_words(text="")
