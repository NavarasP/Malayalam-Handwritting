#!/usr/bin/python
# -*- coding: <UTF-8> -*-

import re
from collections import Counter



def words(text): 
    return re.findall(r'\w+',text)

f = open("words.txt",encoding="utf-8")
vocab = f.read()
f.close()

vocab = vocab.splitlines()


    

def single_edit(word):
    letters    = ' "്  ാ  ി  ീ  ു  ൂ  െ ൃ  െ  ൌ  ം അ ആ ഇ ഉ ഋ എ ഏ ഒ ക ഖ ഗ ഘ ങ ച ഛ ജ ഝ ഞ ട ഠ ഢ ഡ ണ ത ഫ ദ ധ ന പ ഫ ബ ഭ മ യ ര റ ല ള ഴ വ ശ ഷ സ ഹ ൺ ൻ ർ ൽ ൾ ക്ക ക്ഷ ങ്ക ങ്ങ ച്ച ഞ്ച ഞ്ഞ ട്ട ണ്ട ണ്ണ ത്ത ദ്ധ ന്ത ന്ദ ന്ന പ്പ മ്പ മ്മ യ്യ ല്ല ള്ള  ്യ   ്ര  ്വ"'
    letters = letters.split(" ")
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return list(set(deletes + transposes + replaces +inserts))#2 is priority value

def two_letter_edit(words):
        e1 = list(single_edit(words))
        word = []
        for e2 in e1:
            word.append(single_edit(e2))
        return word

def candidates(word):
    candidates = [word]
    if word in vocab:
        return candidates,len(candidates) == 1
    else:
        for i in single_edit(word):
            if i in vocab:
                candidates.append(i)
        for i in two_letter_edit(word):
            if i in vocab:
                candidates.append(i)
        return list(filter(None,candidates)),len(candidates) == 1

def correction(word):
    probable_candidates = candidates(word)
    return probable_candidates
    
