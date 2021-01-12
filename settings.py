import os

from utils.folder_file_manager import make_directory_if_not_exists

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'output'))

PARAGRAPH_EMBED_MODEL = os.path.join(CUR_DIR, 'utils', 'model', 'doc2vec.bin')
MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pruned.word2vec.txt')

KEYWORD_FILE_PATH = "/media/main/Data/Task/DescriptionSimilarityV2/keyword_sample.csv"
SIMILARITY_FILE_PATH = "/media/main/Data/Task/DescriptionSimilarityV2/similarity_sample.csv"
