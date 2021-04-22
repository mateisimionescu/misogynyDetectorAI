import nltk
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def tokenize(text):
    return nltk.TweetTokenizer().tokenize(text)


def get_corpus_vocabulary(corpus):
    counter = Counter()
    for text in corpus:
        text = text.translate(str.maketrans(" "," ", string.punctuation + "‘…❤—“”’"))
        text = text.translate(str.maketrans("", "", string.digits))
        text = re.sub(r'https\S+', '', text)
        tokens = tokenize(text.lower())
        counter.update(tokens)
    return counter


def get_representation(toate_cuvintele, how_many):
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)
    return all_features


def write_prediction(out_file, predictions):
    with open(out_file, 'w') as fout:
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 500)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label'].values

test_data = corpus_to_bow(test_df['text'], wd2idx)

predictii = np.ones(len(test_data))

from sklearn.neighbors import KNeighborsClassifier
clasificator=KNeighborsClassifier(n_neighbors=5)
clasificator.fit(data,labels)

predictii = clasificator.predict(test_data)

write_prediction('sample_submission.csv', predictii)


