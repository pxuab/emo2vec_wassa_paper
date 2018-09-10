import torch
import torch.utils.data as data
from torch.autograd import Variable
import csv
import re
from utils.preprocessing import tokenization, load_vocab, load_vocab_glove
import pickle
import numpy as np
import logging
from models.global_variable import *
# STAND 0 -> very neg, 1 -> neg, 2-> nutral, 3 -> pos, 4 -> very pos
# TUBE 0 -> NEG, 1-> POS
# TUBE 0 -> very neg, 1 -> neg, 2 -> pos, 3 -> very pos
# SEMEVAL 0 -> NEG, 1-> NEUTRAL, 2-> POS


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, vocab, vocab_task, data_type, type_d):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.vocab_task = vocab_task
        self.dataset_selector(data_type, type_d)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.X_data[index], self.X_dataEMB[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

    def flatten(self, tree_sent):
        """
        Flattens constituency trees to get just the tokens.
        """
        label = int(tree_sent[1])
        text = re.sub('\([0-9]', ' ', tree_sent).replace(')', '').split()
        return label, ' '.join(text)

    def read_data_semeval(self, file_name):
        X_data = []
        X_dataEMB = []
        y_data = []
        with open(file_name) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for line in tsvreader:
                
                X_data.append(tokenization(line[3], self.vocab, max_len=40))
                X_dataEMB.append(tokenization(line[3], self.vocab_task, max_len=40))
                if(str(line[2]) == "neutral" or str(line[2]) == "objective-OR-neutral" or str(line[2]) == "objective"):
                    y_data.append(1)
                elif(str(line[2]) == "negative"):
                    y_data.append(0)
                elif(str(line[2]) == "positive"):
                    y_data.append(2)
                else:
                    logging.info("ERROR")
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_openner(self, file_name):
        X_data = []
        X_dataEMB = []
        y_data = []

        with open(file_name + "/neg.txt") as fl:
            for line in fl:
                
                X_data.append(tokenization(line, self.vocab, max_len=10))
                X_dataEMB.append(tokenization(line, self.vocab_task, max_len=10))
                y_data.append(0)

        with open(file_name + "/strneg.txt") as fl:
            for line in fl:
                
                X_data.append(tokenization(line, self.vocab, max_len=10))
                X_dataEMB.append(tokenization(line, self.vocab_task, max_len=10))
                y_data.append(1)

        with open(file_name + "/pos.txt") as fl:
            for line in fl:
                
                X_data.append(tokenization(line, self.vocab, max_len=10))
                X_dataEMB.append(tokenization(line, self.vocab_task, max_len=10))
                y_data.append(2)

        with open(file_name + "/strpos.txt") as fl:
            for line in fl:
                
                X_data.append(tokenization(line, self.vocab, max_len=10))
                X_dataEMB.append(tokenization(line, self.vocab_task, max_len=10))
                y_data.append(3)

        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_sentube(self, file_name):
        X_data = []
        X_dataEMB = []
        y_data = []
        with open(file_name + "/pos.txt") as fl:
            for line in fl:
                
                X_data.append(tokenization(line, self.vocab, max_len=60))
                X_dataEMB.append(tokenization(line, self.vocab_task, max_len=60))
                y_data.append(1)

        with open(file_name + "/neg.txt") as fl:
            for line in fl:
                
                X_data.append(tokenization(line, self.vocab, max_len=60))
                X_dataEMB.append(tokenization(line, self.vocab_task, max_len=60))
                y_data.append(0)
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_sst_coarse(self, file_name):
        X_data = []
        X_dataEMB = []
        y_data = []
        with open(file_name) as fl:
            for line in fl:
                y, X = self.flatten(line)
                if y in [0, 1]:
                    
                    X_data.append(tokenization(X, self.vocab, max_len=40))
                    X_dataEMB.append(tokenization(X, self.vocab_task, max_len=40))
                    y_data.append(0)
                elif y in [3, 4]:
                    
                    X_data.append(tokenization(X, self.vocab, max_len=40))
                    X_dataEMB.append(tokenization(X, self.vocab_task, max_len=40))
                    y_data.append(1)
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_sst_fine(self, file_name):
        X_data = []
        X_dataEMB = []
        y_data = []
        with open(file_name) as fl:
            for line in fl:
                y, X = self.flatten(line)
                
                X_data.append(tokenization(X, self.vocab, max_len=40))
                X_dataEMB.append(tokenization(X, self.vocab_task, max_len=40))
                # 0 -> very neg, 1 -> neg, 2 -> pos, 3 -> very pos
                y_data.append(y)
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_emo(self, file_name, type_d, max_line=1000000):
        vocab_id2word = dict(zip(self.vocab.values(), self.vocab.keys()))
        vocab_task_id2word = dict(zip(self.vocab_task.values(), self.vocab_task.keys()))
        
        if(str(type_d) == "dev"):
            type_d = "valid"
        cnt = 0
        X_data = []
        y_data = []
        X_dataEMB = []
        with open(file_name + "/idx-wang-hashtag-{}.tsv".format(type_d)) as f:
            for line in f.readlines():     
                X_data.append(list(map(int, line.strip().split())))
                X_dataEMB.append([self.vocab_task[vocab_id2word[idx]] if vocab_id2word[idx] in self.vocab_task else self.vocab_task['UNK'] for idx in list(map(int, line.strip().split()))])
                if(cnt > max_line):
                    break
                cnt += 1
        cnt = 0
        with open(file_name + "/clean-wang-hashtag-{}-label.tsv".format(type_d)) as f:
            for line in f.readlines():
                y_data.append(line.strip())
                if(cnt > max_line):
                    break
                cnt += 1
        label_list = list(set(y_data))
        y_data = [label_list.index(l) for l in y_data]
        max_len = 66
        end_idx = self.vocab['PAD']
        X_data = [doc + [end_idx] * (max_len - len(doc)) if len(doc) <
                  max_len else doc[:max_len - 1] + [end_idx] for doc in X_data]
        X_dataEMB = [doc + [self.vocab_task['PAD']] * (max_len - len(doc)) if len(doc) <
                  max_len else doc[:max_len - 1] + [self.vocab_task['PAD']] for doc in X_dataEMB]
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_emo_full(self, file_name, type_d, max_line=1000000):
        vocab_id2word = dict(zip(self.vocab.values(), self.vocab.keys()))
        vocab_task_id2word = dict(zip(self.vocab_task.values(), self.vocab_task.keys()))

        if(str(type_d) == "dev"):
            type_d = "valid"
        cnt = 0
        X_data = [] 
        y_data = []
        X_dataEMB = []
        with open(file_name + "/idx/clean_idx_tweets_{}.pkl".format(type_d),'rb') as f:
            for line in pickle.load(f):
                X_data.append(line)
                X_dataEMB.append([self.vocab_task[vocab_id2word[idx]] if vocab_id2word[idx] in self.vocab_task else self.vocab_task['UNK'] for idx in line])
                if(cnt > max_line):
                    break
                cnt += 1
        cnt = 0
        with open(file_name + "/{}_labels.pkl".format(type_d), 'rb') as f:
            for line in pickle.load(f):
                y_data.append(line)
                if(cnt > max_line):
                    break
                cnt += 1
        label_list = list(set(y_data))
        y_data = [label_list.index(l) for l in y_data]
        max_len = 66
        end_idx = self.vocab['PAD']
        X_data = [doc + [end_idx] * (max_len - len(doc)) if len(doc) <
                  max_len else doc[:max_len - 1] + [end_idx] for doc in X_data]
        X_dataEMB = [doc + [self.vocab_task['PAD']] * (max_len - len(doc)) if len(doc) <
                  max_len else doc[:max_len - 1] + [self.vocab_task['PAD']] for doc in X_dataEMB]
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_SS(self,file_name, type_d):
        X_data = []
        X_dataEMB = []
        y_data = []
        if 'Twitter' in file_name:
            file_name = file_name+'SS-Twitter_'+type_d+'.csv'
        else:
            file_name = file_name+'SS-Youtube_'+type_d+'.csv'
        with open(file_name) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                
                X_data.append(tokenization(line[0], self.vocab, max_len=40))
                X_dataEMB.append(tokenization(line[0], self.vocab_task, max_len=40))
                y_data.append(int(line[1]))

        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data 

    def read_data_wassa(self,file_name, type_d):
        X_data = []
        X_dataEMB = []
        y_data = []        
        file_format ={"train": "%s-ratings-0to1.train.txt",
                    "dev": "%s-ratings-0to1.dev.target.txt",
                    "test": "%s-ratings-0to1.test.target.txt"}
        labels = ["joy", "sadness", "anger", "fear"]
        for e in labels:
            path = file_name + file_format[type_d] % e
            with open(path) as file:
                reader = csv.reader(file, delimiter="\t")
                for line in reader:
                    X_data.append(tokenization(line[1], self.vocab, max_len=60))
                    X_dataEMB.append(tokenization(line[1], self.vocab_task, max_len=60))
                    if line[2] == "joy":
                        y_data.append(0)
                    if line[2] == "sadness":
                        y_data.append(1)
                    if line[2] == "anger":
                        y_data.append(2)
                    if line[2] == "fear":
                        y_data.append(3)
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data 

    def read_data_binary_wassa(self, file_name, type_d):
        X_data = []
        X_dataEMB = []
        y_data = []
        path = file_name+'_'+type_d+'.txt'
        with open(path) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                X_data.append(tokenization(line[0], self.vocab, max_len=60))
                X_dataEMB.append(tokenization(line[0], self.vocab_task, max_len=60))
                y_data.append(int(line[1]))
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data

    def read_data_deepmoji(self, file_name, type_d):
        X_data = []
        X_dataEMB = []
        y_data = []
        file_name = file_name+'_'+type_d+'.txt'
        with open(file_name) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                X_data.append(tokenization(line[0], self.vocab, max_len=300))
                X_dataEMB.append(tokenization(line[0], self.vocab_task, max_len=300))
                y_data.append(int(line[1]))
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data 

    def read_data_personality(self, file_name, type_d):
        X_data = []
        X_dataEMB = []
        y_data = []
        file_name = file_name+'_'+type_d+'.txt'
        with open(file_name) as file:
            for line in file:
                line = line.split("','")
                X_data.append(tokenization(line[0], self.vocab, max_len=300))
                X_dataEMB.append(tokenization(line[0], self.vocab_task, max_len=300))
                y_data.append(int(line[1]))

        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data 

    def read_data_stress(self, file_name, type_d):
        X_data = []
        X_dataEMB = []
        y_data = []        
        path = file_name + "_{}.csv".format(type_d)
        with open(path) as file:
            reader = csv.reader(file, delimiter=",")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                X_data.append(tokenization(line[1], self.vocab, max_len=60))
                X_dataEMB.append(tokenization(line[1], self.vocab_task, max_len=60))
                y_data.append(int(line[2]))
        return torch.LongTensor(X_data), torch.LongTensor(X_dataEMB),y_data 

    def dataset_selector(self, data_type, type_d):
        if(data_type == 'semeval'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_semeval(
                'datasets/semeval/' + type_d + '.tsv')
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'opener'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_openner(
                'datasets/opener/' + type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'tube_auto'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_sentube(
                'datasets/SenTube/auto/' + type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'tube_tablet'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_sentube(
                'datasets/SenTube/tablets/' + type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'sst_coarse'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_sst_coarse(
                'datasets/stanford_sentanalysis/' + type_d + '.txt')
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'sst_fine'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_sst_fine(
                'datasets/stanford_sentanalysis/' + type_d + '.txt')
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'emo'):
        #     from models.global_variable import args
        #     self.X_data, self.X_dataEMB, self.y_data = self.read_data_emo(
        #         'datasets/clean_hashtag/tokenized', type_d, max_line=int(args['max_emo']))
        #     logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
 
            from models.arguments import args
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_emo_full(
                'datasets/hashtag_full_vocab/', type_d, max_line=int(args['max_emo']))
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'wassa'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_wassa(
                'datasets/wassa-2017/', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'SS_T'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/SS-Twitter/SS-Twitter', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'SS_Y'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/SS-Youtube/SS-Youtube', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'isear'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/isear/isear', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'SCv1'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/SCv1/SCv1', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'SCv2'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/SCv2-GEN/SCv2-GEN', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'kaggle'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/kaggle/kaggle', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif(data_type == 'stress'):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_stress(
                'datasets/stress/stress', type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif('SE0714' in data_type):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/SE0714/{}'.format(data_type), type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif('Olympic' in data_type):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/Olympic/{}'.format(data_type), type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif('_personality' in data_type):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_personality(
                'datasets/essay_personality/{}'.format(data_type), type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))    
        elif('abusive' in data_type):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/abusive/{}'.format(data_type), type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif('isear_pos' in data_type):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_deepmoji(
                'datasets/isear/{}'.format(data_type), type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        elif('_wassa' in data_type):
            self.X_data, self.X_dataEMB, self.y_data = self.read_data_binary_wassa(
                'datasets/wassa-2017/binary/{}'.format(data_type), type_d)
            logging.info("{} {} set. SAMPLE:{}".format(data_type,type_d,len(self.X_data)))
        else:
            print("ERROR",data_type)

def get_data(dataset_names, bsz, extra_dim=100):
    logging.info("LOADING DATA")
    # vocab_path = "datasets/clean_hashtag/tokenized/vocab.pkl"
    vocab = load_vocab(vocab_path)
    vocab_task = load_vocab_glove(extra_dim)
    data_loader_TRAIN = {}
    data_loader_DEV = {}
    data_loader_TEST = {}
    for d in dataset_names:
        dataset = Dataset(vocab,vocab_task,data_type=d, type_d='train')
        data_loader_TRAIN[d] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=bsz, shuffle=True)
        dataset = Dataset(vocab,vocab_task,data_type=d, type_d='dev')
        data_loader_DEV[d] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=bsz, shuffle=False)
        dataset = Dataset(vocab,vocab_task,data_type=d, type_d='test')
        data_loader_TEST[d] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=bsz, shuffle=False)
        logging.info("".join(["-" for i in range(100)]))
    return data_loader_TRAIN, data_loader_DEV, data_loader_TEST, len(vocab), vocab


def get_batch(dataset_names, data_loader):
    from models.global_variable import USE_CUDA
    X_batches = {}
    X_batches_EMB = {}    
    y_batches = {}
    for n in dataset_names:
        for d in data_loader[n]:
            if USE_CUDA:
                X_batches[n] = Variable(d[0]).cuda()
                X_batches_EMB[n] = Variable(d[1]).cuda()        
                y_batches[n] = Variable(d[2]).cuda()
            else:
                X_batches[n] = Variable(d[0])
                X_batches_EMB[n] = Variable(d[1])                      
                y_batches[n] = Variable(d[2])
            break
    return X_batches, X_batches_EMB, y_batches

def gen_embeddings(word_dict, dim, in_file=None,
                   init=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    in_file = "trained_models/glove_twitter/glove.twitter.27B.{}d.txt".format(dim)
    num_words = len(word_dict)
    embeddings = np.zeros((num_words, dim))
    logging.info('Embeddings: %d x %d' % (num_words, dim))
    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            pre_trained += 1
            if(len(sp[1:])!= dim):
                logging.info("Warning: One vector is too short. Keep zero for word space")
            else:
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings
