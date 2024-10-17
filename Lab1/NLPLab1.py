#Требуется прочитать текст на русском языке из файла и вывести все пары соседних слов, которые:
#   имеют имена существительные или имена прилагательные на первом или втором месте;
#   совпадают по роду, числу и падежу.
#Все пары следует выводить в виде лемм. Например, если исходная пара имела вид «необычайных университетов», то должна быть выведена пара «необычайный университет».

import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
import pymorphy3

m = pymorphy3.MorphAnalyzer()
f = open(r"d:\Work\4NLP\NLPLab1\text.txt")
string = f.read()

tokens = word_tokenize(string)

for index, token in enumerate(tokens):
    w = m.parse(token)[0]
    if ('NOUN' in w.tag) or ('ADJF' in w.tag):
        wNext = m.parse(tokens[index+1])[0]
        if ((('NOUN' in wNext.tag) or ('ADJF' in wNext.tag)) and (w.tag.gender == wNext.tag.gender) and (w.tag.case == wNext.tag.case) and (w.tag.number == wNext.tag.number)):
            print(w.normal_form, wNext.normal_form)