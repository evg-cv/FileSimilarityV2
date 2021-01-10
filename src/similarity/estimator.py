import os
import pandas as pd
import numpy as np
import ntpath

from sklearn.metrics.pairwise import cosine_similarity
from operator import add
from src.feature.extractor import GFeatureExtractor
from src.semantic.pos_tagging import SemanticAnalyzer
from utils.folder_file_manager import log_print
from settings import COEFFICIENT_A, COEFFICIENT_B, COEFFICIENT_C, COEFFICIENT_D, COEFFICIENT_E, COEFFICIENT_F


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

    def run(self, csv_file_path):

        input_df = pd.read_csv(csv_file_path)
        master_key = input_df["Master"][0]
        master_key_tags = self.semantic_analyzer.extract_pos_tags(text=master_key)
        master_feature = self.extract_feature_from_tags(tag_result=master_key_tags)
        statements = input_df["Text"].values.tolist()
        if statements:
            for s_des in statements:
                try:
                    s_des_tags = self.semantic_analyzer.extract_pos_tags(text=s_des)
                    s_des_feature = self.extract_feature_from_tags(tag_result=s_des_tags)
                    proximity = cosine_similarity([master_feature], [s_des_feature])
                    print(f"{s_des}: {self.analyze_modeling(value=float(proximity[0][0]))}")
                except Exception as e:
                    log_print(e)

        else:
            print("[INFO] There are not any statements to estimate in")

        return


if __name__ == '__main__':
    SemanticSimilarity().run(csv_file_path="/media/main/Data/Task/DescriptionSimilarityV2/test.csv")
