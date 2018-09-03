import collections
import mojimoji
import io
import re
import numpy
from numpy import random
from Const import UNK_ID, BOS_ID, EOS_ID

digit_pattern = re.compile(r'(\d( \d)*)+')

def open_file(path):
    return io.open(path, encoding='utf-8', errors='ignore')

def load_file(path):
    with open_file(path) as f:
        for line in f:
            line = mojimoji.zen_to_han(line, kana=False)
            line = digit_pattern.sub('#', line)
            words = line.rstrip().split(' ')
            yield words

def get_vocab(path, vocabsize, minfreq):
    counter = collections.Counter()
    for words in load_file(path):
        for w in words:
            counter[w] += 1
    vocab = [w for w, f in counter.most_common(vocabsize) if f >= minfreq]
    return vocab

def save_vocab(path, vocab):
    with open(path, 'w') as f:
        for w in vocab:
            f.write(w)
            f.write('\n')

def word2id(path, vocab):
    w2id= {w:i+3 for i, w in enumerate(vocab)}
    w2id['BOS'] = BOS_ID
    w2id['EOS'] = EOS_ID
    data = []
    for words in load_file(path):
        seq = ([w2id.get(w, UNK_ID) for w in words])
        data.append(numpy.array(seq, dtype=numpy.int32))
    return data
