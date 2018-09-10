from hltc_preprocess.tweets import tokenize_tweets, clean_tweet
import pickle
import numpy as np
import torch 

padding = 'PAD'
unknown = 'UNK'

def load_vocab(vocab_path):
    """ load vocab file with pickle format """
    with open(vocab_path, 'rb') as f:
        word2id = pickle.load(f, encoding='utf-8')['word2id']
    if padding not in word2id:
        word2id[padding] = len(word2id)
    return word2id

def load_vocab_glove(dim):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    in_file = "trained_models/glove_twitter/glove.twitter.27B.{}d.txt".format(dim)

    word2index = {'UNK':0,'PAD':1}
    i = 2
    for line in open(in_file).readlines():
        sp = line.split()
        word2index[sp[0]] = i
        i += 1
    return word2index

def tokenization(string, vocab, max_len=20, return_tokens=False):
    """ tokenize one string to idxs """
    string = clean_tweet(string)
    string = [string]
    tokens = tokenize_tweets(string, segment=False)
 
    idx_tokens = [vocab[w] if w in vocab else vocab['UNK'] for w in tokens[0]]
    tokens = [w if w in vocab else 'UNK' for w in tokens[0]]  

    if len(idx_tokens) < max_len:
        idx_tokens += (max_len - len(idx_tokens)) * [vocab[padding]]

    idx_tokens = idx_tokens[:max_len]
    if return_tokens:
        return idx_tokens, tokens
    else:
        return idx_tokens

if __name__ == "__main__":
    vocab_path = "../datasets/clean_hashtag/tokenized/vocab.pkl"
    vocab = load_vocab(vocab_path)
    print(tokenization("I love ...you", vocab, return_tokens=True))
