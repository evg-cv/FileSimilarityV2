import spacy

from pycorenlp import StanfordCoreNLP
from spacy.lemmatizer import VERB
from utils.const import POS_TAGSET


class SemanticAnalyzer:
    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.lemmatizer = spacy.load('en_core_web_sm').vocab.morphology.lemmatizer

    @staticmethod
    def get_tagged_words(tag_result):
        tagged_words = []
        for t_result in tag_result:
            t_result = t_result.replace("(", "").replace(")", "").replace(".", "")
            for pos_tag in POS_TAGSET:
                t_result = t_result.replace(pos_tag, "")
            t_result_tagged_words = []
            for t_s_result in t_result.split(" "):
                if t_s_result != "":
                    t_result_tagged_words.append(t_s_result)
            tagged_words += t_result_tagged_words

        return tagged_words

    def extract_pos_tags(self, text):

        output = self.nlp.annotate(text, properties={
            'annotators': 'parse',
            'outputFormat': 'json'
        })

        pos_tag_results = output['sentences']

        doc_pos_result = []
        for sent_tag_result in pos_tag_results:
            sentence_pos_result = {"subject": [], "verb": [], "object": []}
            structured_tags = sent_tag_result['parse'].split("\n")
            if structured_tags[1] == "  (S":
                subject_results = []
                verb_results = []
                tag_index = 2
                passive_ret = False
                while "VP" not in structured_tags[tag_index]:
                    subject_results.append(structured_tags[tag_index])
                    tag_index += 1
                while "NP" not in structured_tags[tag_index]:
                    vbn_word = ""
                    if "VBN" in structured_tags[tag_index]:
                        passive_ret = True
                        vbn_word_index = structured_tags[tag_index].index("VBN") + 4
                        vbn_character = structured_tags[tag_index][vbn_word_index]
                        while vbn_character != ")":
                            vbn_word += vbn_character
                            vbn_word_index += 1
                            vbn_character = structured_tags[tag_index][vbn_word_index]
                        structured_tags[tag_index] = \
                            structured_tags[tag_index].replace(vbn_word, self.lemmatizer(vbn_word, VERB)[0])
                    if passive_ret:
                        if vbn_word != "":
                            verb_results = []
                        if not verb_results:
                            verb_results = self.lemmatizer(vbn_word, VERB)
                    else:
                        verb_results.append(structured_tags[tag_index])
                    tag_index += 1
                    if tag_index == len(structured_tags):
                        tag_index -= 1
                        break
                object_results = structured_tags[tag_index:]
                if not passive_ret:
                    sentence_pos_result["subject"] = self.get_tagged_words(tag_result=subject_results)
                    sentence_pos_result["object"] = self.get_tagged_words(tag_result=object_results)
                else:
                    sentence_pos_result["subject"] = self.get_tagged_words(tag_result=object_results)
                    sentence_pos_result["object"] = self.get_tagged_words(tag_result=subject_results)
                sentence_pos_result["verb"] = self.get_tagged_words(tag_result=verb_results)

            doc_pos_result.append(sentence_pos_result)

        return doc_pos_result


if __name__ == '__main__':
    SemanticAnalyzer().extract_pos_tags(text="")
