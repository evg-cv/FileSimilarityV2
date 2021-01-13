import os
import time
import pandas as pd
import random
from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity


from settings import PARAGRAPH_EMBED_MODEL
# from utils.logger import logger


class ParagraphEmbed:
    def __init__(self):
        """
        # Load pre-trained paragraph vector model
        https://github.com/jhlau/doc2vec
            - English Wikipedia DBOW (1.4GB)
            - Associated Press News DBOW (0.6GB)
        """

        # model = "toy_data/model.bin"
        # logger.info(f"load doc2vec model...")
        paragraph_embed_model_file = PARAGRAPH_EMBED_MODEL

        # logger.info(f"\tdoc2vec model: {os.path.split(paragraph_embed_model_file)[1]}")
        self.doc2vec_model = Doc2Vec.load(paragraph_embed_model_file)

        # inference hyper-parameters
        self.start_alpha = 0.025
        self.infer_epoch = 20

        # self.update_feature_identifier()

    @staticmethod
    def _tagcol_paragraph_embeddings_features(train_data):
        """
            Input: a collection of columns stored in a dataframe column 'values'
            Output: tagged columns.
            Only needed for training.

        :param train_data:
        :return:
        """

        # Expects a dataframe with a 'values' column
        train_data_values = train_data['values']
        random.seed(round(time.time()))

        columns = [TaggedDocument(random.sample(col, min(MAX_COL_LEN, len(col))), [i])
                   for i, col in enumerate(train_data_values.values)]

        return columns

    @staticmethod
    def _train_paragraph_embeddings_features(columns, dim=400):
        """
            Input: returned tagged document collection from tagcol_paragraph_embeddings_features
            Output: a stored retrained model
            Only needed for training.
        :param columns:
        :return:
        """

        # Train Doc2Vec model
        model = Doc2Vec(columns, dm=0, negative=3, workers=8, vector_size=dim, epochs=20, min_count=2, seed=13)

        # Save trained model
        model_file = '../sherlock/features/par_vec_retrained_{}.pkl'.format(dim)
        model.save(model_file)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    def train(self, train_data):
        columns = self._tagcol_paragraph_embeddings_features(train_data=train_data)

        self._train_paragraph_embeddings_features(columns=columns)

    def infer_paragraph_embeddings_features(self, data):
        """
            Input: a single column in the form of a pandas Series.
            Output: ordered dictionary holding paragraph vector features

        :param data:
        :return:
        """

        # f = OrderedDict()

        # Infer paragraph vector for data sample
        vec = self.doc2vec_model.infer_vector(data, steps=self.infer_epoch, alpha=self.start_alpha)
        # for i in range(self.doc2vec_model.vector_size):
        #     f[f'par_vec_{i}'] = vec[i]

        return vec

    def zero_feature(self):
        f = OrderedDict()
        for i in range(self.doc2vec_model.vector_size):
            f[f'par_vec_{i}'] = 0
        return f

    def update_feature_identifier(self):
        feature_identifier_file = os.path.join(FEATURE_IDENTIFIERS_DIR, 'par_col.tsv')

        feature_names = []
        for i in range(self.doc2vec_model.vector_size):
            feature_names.append(f'par_vec_{i}')

        df = pd.DataFrame({"feature_names": feature_names})
        df.to_csv(feature_identifier_file, sep="\t", index=True, header=False)
        return True


if __name__ == '__main__':
    paragraph = ParagraphEmbed()
    master_vec = paragraph.infer_paragraph_embeddings_features(data=[])
    text_iters = []
    for t_iter in text_iters:
        t_vec = paragraph.infer_paragraph_embeddings_features(data=[t_iter])
        similarity = cosine_similarity([master_vec], [t_vec])
        print(f"Similarity with {t_iter}: {similarity}")
