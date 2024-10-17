import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
import pymorphy3

m = pymorphy3.MorphAnalyzer()
f = open(r"d:\Work\4NLP\NLPLab1\text.txt")
string = f.read()

tokens = word_tokenize(string)
w = m.parse("Британских")[0]
print(w.tag)
w = m.parse("Дивизий")[0]
print(w.tag)
#for index, token in enumerate(tokens):
#    w = m.parse(token)[0]
#    if ('NOUN' in w.tag) or ('ADJF' in w.tag):
#        wNext = m.parse(tokens[index+1])[0]
#        if ((('NOUN' in wNext.tag) or ('ADJF' in wNext.tag)) and (w.tag.gender == wNext.tag.gender) and (w.tag.case == wNext.tag.case) and (w.tag.number == wNext.tag.number)):
#            print(w.normal_form, wNext.normal_form)