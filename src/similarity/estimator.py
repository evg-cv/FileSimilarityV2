import os
import pandas as pd
import ntpath
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from src.feature.extractor import GFeatureExtractor
from utils.folder_file_manager import log_print
from settings import INPUT_EXCEL_PATH, SIMILARITY_NUMBER, SIMILARITY_THRESH, OUTPUT_DIR


class DescriptionSimilarity:
    def __init__(self):
        self.feature_extractor = GFeatureExtractor()

    def run(self):
        control_similarities = []
        risk_similarities = []
        control_similarity_values = []
        similarity_rations = []
        risk_similarity_values = []
        avg_similarity_values = []

        file_name = ntpath.basename(INPUT_EXCEL_PATH).replace(".xlsx", "")
        output_file_path = os.path.join(OUTPUT_DIR, f"{file_name}_result.csv")

        input_df = pd.read_excel(INPUT_EXCEL_PATH)
        risk_descriptions = input_df["Risk Description"].values.tolist()
        control_descriptions = input_df["Control Description"].values.tolist()
        control_features = []
        for c_des in control_descriptions:
            try:
                c_des_feature = self.feature_extractor.get_feature_token_words(text=c_des)
            except Exception as e:
                c_des_feature = None
                log_print(e)
            control_features.append(c_des_feature)

        for i, c_i_feature in enumerate(control_features):
            i_similarity = []
            if c_i_feature is not None:
                for j, c_j_feature in enumerate(control_features):
                    if j == i or c_j_feature is None:
                        continue
                    i_j_similarity = cosine_similarity([c_i_feature], [c_j_feature])
                    if i_j_similarity[0][0] >= SIMILARITY_THRESH:
                        i_similarity.append([j, i_j_similarity[0][0]])

            if not i_similarity:
                control_similarities.append("NA")
                risk_similarities.append("NA")
                control_similarity_values.append("NA")
                similarity_rations.append("NA")
                risk_similarity_values.append("NA")
                avg_similarity_values.append("NA")
            else:
                sorted_similarity = sorted(i_similarity, key=lambda k: k[1], reverse=True)[:SIMILARITY_NUMBER]
                similarity_indices = np.array(sorted_similarity)[:, 0].astype(np.int)
                init_controls = ""
                init_risks = ""
                init_control_scores = ""
                init_rations = ""
                init_risk_scores = ""
                init_avg_scores = ""
                for m, s_index in enumerate(similarity_indices):
                    risk_des_feature = self.feature_extractor.get_feature_token_words(text=risk_descriptions[i])
                    risk_similar_feature = \
                        self.feature_extractor.get_feature_token_words(text=risk_descriptions[s_index])
                    risk_similarity = cosine_similarity([risk_des_feature], [risk_similar_feature])
                    init_controls += control_descriptions[s_index] + ","
                    init_risks += str(risk_descriptions[s_index]) + ","
                    init_control_scores += str(sorted_similarity[m][1]) + ","
                    init_risk_scores += str(risk_similarity[0][0]) + ","
                    init_avg_scores += str(0.5 * (risk_similarity[0][0] + sorted_similarity[m][1])) + ","
                    if sorted_similarity[m][1] >= 0.75:
                        init_rations += "high" + ","
                    elif 0.5 < sorted_similarity[m][1] < 0.75:
                        init_rations += "medium" + ","
                    else:
                        init_rations += "low" + ","
                control_similarities.append(init_controls[:-1])
                risk_similarities.append(init_risks[:-1])
                control_similarity_values.append(init_control_scores[:-1])
                similarity_rations.append(init_rations[:-1])
                risk_similarity_values.append(init_risk_scores)
                avg_similarity_values.append(init_avg_scores)

            print(f"Processed Control Description {i + 1} rows")

        input_df["Similar Sentences"] = control_similarities
        input_df["Risk Sentences"] = risk_similarities
        input_df["Similar Values"] = control_similarity_values
        input_df["Similar Rations"] = similarity_rations
        input_df["Risk Values"] = risk_similarity_values
        input_df["Average Values"] = avg_similarity_values

        input_df.to_csv(output_file_path, index=True, header=True, mode="w")

        print(f"[INFO] Successfully saved in {output_file_path}")

        return


if __name__ == '__main__':
    DescriptionSimilarity().run()
