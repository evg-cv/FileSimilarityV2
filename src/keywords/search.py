from sklearn.metrics.pairwise import cosine_similarity
from src.feature.extractor import GFeatureExtractor
from src.feature.tokenizer import TextPreprocessor


class KeywordSearcher:
    def __init__(self):
        self.feature_extractor = GFeatureExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.result = {"Founded Keyword": [], "Synonyms": []}

    def preprocess_text(self, text, master_ret=False):
        text_words = []
        sentences = self.text_preprocessor.tokenize_sentence(text=text)
        for sent in sentences:
            token_words = self.text_preprocessor.tokenize_word(sample=sent.text, master_ret=master_ret)
            text_words += token_words

        return text_words

    def search_availability_keyword(self, keywords, text_iterations):
        print(f"[INFO] Step1 Availability of Keywords Processing... ")
        available_keywords = []
        not_available_keywords = []
        regenerated_keywords = []
        for keyword in keywords:
            regenerated_keyword = ' '.join(map(str, self.preprocess_text(text=keyword)))
            regenerated_keywords.append(regenerated_keyword)
        for t_iteration in text_iterations:
            regenerated_text = ' '.join(map(str, self.preprocess_text(text=t_iteration)))
            for regenerated_keyword, keyword in zip(regenerated_keywords, keywords):
                if regenerated_keyword in regenerated_text and keyword not in available_keywords:
                    available_keywords.append(keyword)

        for keyword, regenerated_keyword in zip(keywords, regenerated_keywords):
            if keyword not in available_keywords:
                not_available_keywords.append(regenerated_keyword)

        self.result["Founded Keyword"] = available_keywords
        print(f"[INFO] Step1 Availability of Keywords Finished ")

        return not_available_keywords

    def extract_synonyms(self, search_words, text_iterations):
        print(f"[INFO] Step2 Synonyms of Keywords Processing...")
        text_words = []
        text_word_features = []
        for t_iteration in text_iterations:
            text_words += self.preprocess_text(text=t_iteration, master_ret=True)
        for t_word in text_words:
            text_word_features.append(self.feature_extractor.get_feature_token_words(text=t_word))
        for s_word in search_words:
            if len(s_word.split(" ")) > 1:
                self.result["Synonyms"].append(f"{s_word}: Not Word!")
                continue
            s_word_feature = self.feature_extractor.get_feature_token_words(text=s_word)
            synonym_score = []
            for t_word, t_word_feature in zip(text_words, text_word_features):
                proximity = cosine_similarity([s_word_feature], [t_word_feature])
                synonym_score.append(proximity[0][0])
            self.result["Synonyms"].append(f"{s_word}: {text_words[synonym_score.index(max(synonym_score))]}!")

        print(f"[INFO] Step2 Synonyms of Keywords Finished...")

        return

    def extract_beyond_master(self, master_keys, text_iterations):
        print(f"[INFO] Step3 Content beyond MasterKey Processing...")
        master_key_words = []
        for master_key in master_keys:
            master_key_words += self.preprocess_text(text=master_key, master_ret=True)
        for i, t_iteration in enumerate(text_iterations):
            t_iteration_words = self.preprocess_text(text=t_iteration, master_ret=True)
            self.result[f"Words not in {i + 1}th iteration"] = []
            for t_i_word in t_iteration_words:
                if t_i_word not in master_key_words and \
                        t_i_word not in self.result[f"Words not in {i + 1}th iteration"]:
                    self.result[f"Words not in {i + 1}th iteration"].append(t_i_word)

        print(f"[INFO] Step3 Content beyond MasterKey Finished...")

        return

    def run(self, keywords, master_keys, text_iterations):
        not_available_keywords = self.search_availability_keyword(keywords=keywords, text_iterations=text_iterations)
        self.extract_synonyms(search_words=not_available_keywords, text_iterations=text_iterations)
        self.extract_beyond_master(master_keys=master_keys, text_iterations=text_iterations)

        return self.result


if __name__ == '__main__':
    import pandas as pd
    from utils.tool import check_text
    from settings import KEYWORD_FILE_PATH, SIMILARITY_FILE_PATH

    keywords_ = check_text(str_list=pd.read_csv(KEYWORD_FILE_PATH)["Keyword"].values.tolist())
    master_keys_ = check_text(str_list=pd.read_csv(SIMILARITY_FILE_PATH)["Master Key"].values.tolist())
    text_iterations_ = check_text(str_list=pd.read_csv(SIMILARITY_FILE_PATH)["Text Iteration"].values.tolist())

    KeywordSearcher().run(keywords=keywords_, text_iterations=text_iterations_, master_keys=master_keys_)
