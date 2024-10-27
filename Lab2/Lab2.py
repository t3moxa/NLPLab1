#Используя import gensim, необходимо реализовать вычисление десяти самых близких по смыслу слов,
# находящихся в окрестности от результата операций сложения и вычитания в векторной модели.
#Каждому студенту преподавателем будет дана пара слов и необходимо найти такую линейную комбинацию исходных слов,
# чтобы в результате вычислений заданная пара попадала в первую десятку.
# Процесс, Регуляция
import re
import gensim

pat = re.compile("(.*)_NOUN")

#pos = ["система_NOUN","технология_NOUN", "правило_NOUN", "производство_NOUN", "время_NOUN", "контроль_NOUN", "режим_NOUN"]
#neg = ["предприятие_NOUN"]
pos = ["жизнедеятельность_NOUN", "функционирование_NOUN", "время_NOUN"]
#neg = ["производство_NOUN"]
word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)
dist = word2vec.most_similar(
    positive=pos,
    #negative=neg
    )
print(
    pos,
    #neg
    )
for i in dist:
    e = pat.match(i[0])
    if e is not None:
        print(e.group(1))
