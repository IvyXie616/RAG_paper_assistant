import jieba
import re

cn = "人工智能的，前途无量。"
en = "人工智能的前途无量。Artificial intelligence has a very promising future."

words_cn = re.findall(r"\b\w+\b", cn.lower())
chinese_cn = list(jieba.cut(cn))

words_en = re.findall(r"\b\w+\b", en.lower())
chinese_en = list(jieba.cut(en))

print(words_cn)
print(chinese_cn)
print(words_en)
print(chinese_en)