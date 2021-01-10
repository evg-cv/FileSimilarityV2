# Import dependencies
import pandas as pd
import texthero as hero
from texthero import preprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def get_clean_text(text):
    custom_pipeline = [preprocessing.fillna,
                       # preprocessing.lowercase,
                       preprocessing.remove_whitespace,
                       preprocessing.remove_diacritics
                       # preprocessing.remove_brackets
                       ]
    clean_text = hero.clean(text, custom_pipeline)
    clean_text = [n.replace('{', '') for n in clean_text]
    clean_text = [n.replace('}', '') for n in clean_text]
    clean_text = [n.replace('(', '') for n in clean_text]
    clean_text = [n.replace(')', '') for n in clean_text]

    return clean_text


def create_doc2vec(clean_text):
    card_docs = [TaggedDocument(doc.split(' '), [i]) for i, doc in enumerate(clean_text)]
    # model = Doc2Vec(vector_size=64, min_count=1, epochs=20)
    # instantiate model
    model = Doc2Vec(vector_size=64, window=2, min_count=1, workers=8, epochs=40)
    # build vocab
    model.build_vocab(card_docs)
    # train model
    model.train(card_docs, total_examples=model.corpus_count, epochs=model.epochs)
    # generate vectors
    card2vec = [model.infer_vector((clean_text[i].split(' ')))
                for i in range(0, len(clean_text))]
    # Create a list of lists
    dtv = np.array(card2vec).tolist()

    return dtv


def create_hub_vec(clean_text):
    embeddings = embed(clean_text)
    # create list from np arrays
    use = np.array(embeddings).tolist()
    # add lists as dataframe column
    return [v for v in use]


def calculate_similarities():
    df = pd.read_csv("/media/main/Data/Task/DescriptionSimilarityV2/test.csv")

    text_iterations = df["Text"]
    print(f"[INFO] Master Key: {text_iterations[0]}")

    df["text_clean_iterations"] = get_clean_text(text=text_iterations)

    ti_idf_text_iterations = hero.tfidf(df["text_clean_iterations"], max_features=3000)

    print("[INFO] TI-IDF case:")
    estimate_similarity(iter_vec=ti_idf_text_iterations, iter_text=text_iterations)

    text_iter_doc2vec = create_doc2vec(clean_text=df["text_clean_iterations"])

    print("[INFO] Doc2Vec case:")
    estimate_similarity(iter_vec=text_iter_doc2vec, iter_text=text_iterations)

    text_iter_hub_vec = create_hub_vec(clean_text=df["text_clean_iterations"])

    print("[INFO] Hub case:")
    estimate_similarity(iter_vec=text_iter_hub_vec, iter_text=text_iterations)

    return


def estimate_similarity(iter_vec, iter_text):
    master_vec = iter_vec[0]
    for i_vec, i_text in zip(iter_vec[1:], iter_text[1:]):
        similarity = cosine_similarity([master_vec], [i_vec])
        print(f"Similarity with {i_text}: {similarity}")

    return


if __name__ == '__main__':
    calculate_similarities()
