from gensim.models import KeyedVectors
from pprint import pprint

model = KeyedVectors.load_word2vec_format('enhanced_vectors.bin', binary=True)
# model = KeyedVectors.load_word2vec_format('light_Tencent_AILab_ChineseEmbedding.bin', binary=True)

list = model.most_similar("抛弃", topn=15)
pprint(list)

# similarity = model.similarity("奇怪", "怪哉")
# pprint(f"相似度：{similarity:.4f}")