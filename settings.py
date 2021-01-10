import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

PARAGRAPH_EMBED_MODEL = os.path.join(CUR_DIR, 'utils', 'model', 'doc2vec.bin')
MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pruned.word2vec.txt')

POS_TAGSET = ["WHADVP", "SBARQ", "SBAR", "SINV", "WHNP", "WHPP", "ADJP", "ADVP", "PRP$", "JJR", "JJS", "NNS", "NNP",
              "NNPS", "PDT", "POS", "PRP", "PP$", "RBR", "RBS", "SYM", "VBD", "VBN", "LS", "MD", "NN", "RB", "RP", "NP",
              "PP", "SQ", "VP", "CC", "CD", "DT", "EX", "FW", "IN", "JJ"]

COEFFICIENT_A = 1795.92
COEFFICIENT_B = -10853.27
COEFFICIENT_C = 26098.96
COEFFICIENT_D = -31220.01
COEFFICIENT_E = 18582.72
COEFFICIENT_F = -4403.33
