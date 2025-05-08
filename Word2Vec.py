from gensim.models import KeyedVectors
from pprint import pprint
import time

loadtime1 = time.time()
model = KeyedVectors.load_word2vec_format('light_Tencent_AILab_ChineseEmbedding.bin', binary=True)
loadtime = time.time() - loadtime1
print(f"加载模型时间：{loadtime:.2f}秒")

simtime1 = time.time()
similarity = model.similarity("一", "一个")
simtime = time.time() - simtime1
pprint(f"相似度：{similarity:.4f}")
print(f"计算相似度时间：{simtime:.2f}秒")

similarity = model.similarity("抛弃", "放弃")
pprint(f"相似度：{similarity:.4f}")
#
# list = model.most_similar("错误", topn=10)
# pprint(list)