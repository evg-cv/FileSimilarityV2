import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from operator import add
from src.feature.extractor import GFeatureExtractor
from src.semantic.pos_tagging import SemanticAnalyzer
from utils.folder_file_manager import log_print
from utils.const import COEFFICIENT_A, COEFFICIENT_B, COEFFICIENT_C, COEFFICIENT_D, COEFFICIENT_E, COEFFICIENT_F


class SemanticSimilarity:
    def __init__(self):
        self.feature_extractor = GFeatureExtractor()
        self.semantic_analyzer = SemanticAnalyzer()

    @staticmethod
    def analyze_modeling(value):
        return COEFFICIENT_A + COEFFICIENT_B * value + COEFFICIENT_C * value ** 2 + COEFFICIENT_D * value ** 3 + \
               COEFFICIENT_E * value ** 4 + COEFFICIENT_F * value ** 5

    def extract_feature_from_tags(self, tag_result):
        tag_feature = np.zeros(900)
        for sent_result in tag_result:
            sent_feature = np.concatenate((self.feature_extractor.get_feature_token_words(text=sent_result["subject"]),
                                           self.feature_extractor.get_feature_token_words(text=sent_result["verb"]),
                                           self.feature_extractor.get_feature_token_words(text=sent_result["object"])),
                                          axis=0)
            tag_feature = list(map(add, tag_feature, sent_feature))

        return tag_feature

    def run(self, master_keys, text_iterations):

        print(f"[INFO] Step4 NLP Text Semantic Similarity Estimation Processing...")

        result = {"Text/Master": [f"Text Iteration {i + 1}" for i in range(len(text_iterations))]}
        text_features = []
        for t_iteration in text_iterations:
            t_iter_tags = self.semantic_analyzer.extract_pos_tags(text=t_iteration)
            t_iter_feature = self.extract_feature_from_tags(tag_result=t_iter_tags)
            text_features.append(t_iter_feature)

        for i, master_key in enumerate(master_keys):
            master_key_tags = self.semantic_analyzer.extract_pos_tags(text=master_key)
            master_feature = self.extract_feature_from_tags(tag_result=master_key_tags)
            result[f"Master Key {i + 1}"] = []
            for t_feature in text_features:
                try:
                    proximity = cosine_similarity([master_feature], [t_feature])
                    result[f"Master Key {i + 1}"].append(self.analyze_modeling(value=float(proximity[0][0])))
                except Exception as e:
                    log_print(e)

        print(f"[INFO] Step4 NLP Text Semantic Similarity Estimation Finished...")

        return result


if __name__ == '__main__':
    import pandas as pd
    from utils.tool import check_text
    from settings import SIMILARITY_FILE_PATH

    master_keys_ = check_text(str_list=pd.read_csv(SIMILARITY_FILE_PATH)["Master Key"].values.tolist())
    text_iterations_ = check_text(str_list=pd.read_csv(SIMILARITY_FILE_PATH)["Text Iteration"].values.tolist())

    SemanticSimilarity().run(master_keys=master_keys_, text_iterations=text_iterations_)
