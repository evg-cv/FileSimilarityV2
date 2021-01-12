import os
import ntpath
import pandas as pd

from src.keywords.search import KeywordSearcher
from src.similarity.estimator import SemanticSimilarity
from utils.tool import check_text
from settings import KEYWORD_FILE_PATH, SIMILARITY_FILE_PATH, OUTPUT_DIR


def save_result(file_path, result):
    df_list = []
    for res_key in result.keys():
        df_list.append(result[res_key])
    headers = list(result.keys())
    output_df = pd.DataFrame(df_list).T
    output_df.to_csv(file_path, index=False, header=headers, mode='w')
    print(f"[INFO] Successfully saved into {file_path}")

    return


def run():
    input_file_name = ntpath.basename(SIMILARITY_FILE_PATH).replace(".csv", "")
    search_file_path = os.path.join(OUTPUT_DIR, f"{input_file_name}_keyword_result.csv")
    similarity_file_path = os.path.join(OUTPUT_DIR, f"{input_file_name}_similarity_result.csv")
    keywords = check_text(str_list=pd.read_csv(KEYWORD_FILE_PATH)["Keyword"].values.tolist())
    master_keys = check_text(str_list=pd.read_csv(SIMILARITY_FILE_PATH)["Master Key"].values.tolist())
    text_iterations = check_text(str_list=pd.read_csv(SIMILARITY_FILE_PATH)["Text Iteration"].values.tolist())
    search_result = KeywordSearcher().run(keywords=keywords, master_keys=master_keys, text_iterations=text_iterations)
    similarity_result = SemanticSimilarity().run(master_keys=master_keys, text_iterations=text_iterations)
    save_result(file_path=search_file_path, result=search_result)
    save_result(file_path=similarity_file_path, result=similarity_result)

    return


if __name__ == '__main__':
    run()
