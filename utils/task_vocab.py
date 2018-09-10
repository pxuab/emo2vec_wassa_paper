from collections import Counter
from hltc_preprocess.tweets import tokenize_tweets, clean_tweet
import re
import csv
import pickle
import numpy as np

def cleaner(string):
    string = clean_tweet(string)
    string = [string]
    tokens = tokenize_tweets(string, segment=False)
    return tokens

def flatten( tree_sent):
    """
    Flattens constituency trees to get just the tokens.
    """
    label = int(tree_sent[1])
    text = re.sub('\([0-9]', ' ', tree_sent).replace(')', '').split()
    return label, ' '.join(text)

def read_data_semeval(file_name):
    with open(file_name) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            token = cleaner(line[3])
            index_words(token)


def read_data_openner( file_name):
    with open(file_name + "/neg.txt") as fl:
        for line in fl:
            token = cleaner(line)
            index_words(token)

    with open(file_name + "/strneg.txt") as fl:
        for line in fl:
            token = cleaner(line)
            index_words(token)

    with open(file_name + "/pos.txt") as fl:
        for line in fl:
            token = cleaner(line)
            index_words(token)

    with open(file_name + "/strpos.txt") as fl:
        for line in fl:
            token = cleaner(line)
            index_words(token)

def read_data_sentube( file_name):
    with open(file_name + "/pos.txt") as fl:
        for line in fl:
            token = cleaner(line)
            index_words(token)

    with open(file_name + "/neg.txt") as fl:
        for line in fl:
            token = cleaner(line)
            index_words(token)

def read_data_sst_coarse( file_name):
    with open(file_name) as fl:
        for line in fl:
            _, X = flatten(line)
            token = cleaner(X)
            index_words(token)

def read_data_sst_fine( file_name):
    with open(file_name) as fl:
        for line in fl:
            _, X = flatten(line)
            token = cleaner(X)
            index_words(token)

def read_data_SS(file_name, type_d):
    if 'Twitter' in file_name:
        file_name = file_name+'SS-Twitter_'+type_d+'.csv'
    else:
        file_name = file_name+'SS-Youtube_'+type_d+'.csv'
    with open(file_name) as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:
            token = cleaner(line[0])
            index_words(token)

def read_data_wassa(file_name, type_d):   
    file_format ={"train": "%s-ratings-0to1.train.txt",
                "dev": "%s-ratings-0to1.dev.target.txt",
                "test": "%s-ratings-0to1.test.target.txt"}
    labels = ["joy", "sadness", "anger", "fear"]
    for e in labels:
        path = file_name + file_format[type_d] % e
        with open(path) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                token = cleaner(line[1])
                index_words(token)

def read_data_isear( file_name, type_d):   
    path = file_name + "isear_{}_resplit.csv".format(type_d)
    with open(path) as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:
            token = cleaner(line[0])
            index_words(token)

def dataset_selector( data_type, type_d):
    if(data_type == 'semeval'):
        read_data_semeval('datasets/semeval/' + type_d + '.tsv')
    elif(data_type == 'opener'):
        read_data_openner('datasets/opener/' + type_d)
    elif(data_type == 'tube_auto'):
        read_data_sentube('datasets/SenTube/auto/' + type_d)
    elif(data_type == 'tube_tablet'):
        read_data_sentube('datasets/SenTube/tablets/' + type_d)
    elif(data_type == 'sst_coarse'):
        read_data_sst_coarse('datasets/stanford_sentanalysis/' + type_d + '.txt')
    elif(data_type == 'sst_fine'):
        read_data_sst_fine('datasets/stanford_sentanalysis/' + type_d + '.txt')
    elif(data_type == 'SS_T'):
        read_data_SS('datasets/SS-Twitter/', type_d)
    elif(data_type == 'SS_Y'):
        read_data_SS('datasets/SS-Youtube/', type_d)
    elif(data_type == 'wassa'):
        read_data_wassa('datasets/wassa-2017/', type_d)
    elif(data_type == 'isear'):
        read_data_isear('datasets/isear/', type_d)

word2count = Counter()
def index_words(sentence):
    for word in sentence[0]:
        word2count[word] += 1

dataset_names = ["isear","wassa","SS_T","SS_Y","semeval", "opener", "tube_auto",
                 "tube_tablet", "sst_coarse", "sst_fine"]
for d in dataset_names:
    dataset_selector(d,"train")

ls = word2count.most_common(10000000)
print('#Words: %d -> %d' % (len(word2count), len(ls)))
for key in ls[:10]:
    print(key)
print('...')
for key in ls[-10:]:
    print(key)

word2index = {w[0]: index + 2 for (index, w) in enumerate(ls)}
word2index["UNK"] = 0
word2index["PAD"] = 1


# with open('datasets/vocab_task.pickle', 'wb') as handle:
#     pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
embedding_dim = 100


