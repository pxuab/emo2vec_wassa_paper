import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
from tqdm import tqdm
from models.global_variable import emo2vec_file, dataset2class,use_acc_dict
from sklearn.metrics import f1_score
from utils.data_loader import gen_embeddings
import numpy as np

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dr=0.0,bidirectional=True, pretrained=True, allow_tuning_emb=True, glove_embedding=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.emo2vec = nn.Embedding(len(input_size), hidden_size, padding_idx=len(input_size) - 1)
        self.allow_tuning_emb = allow_tuning_emb
        self.glove_embedding = glove_embedding

        if(pretrained):
            print("LOADING PRETRAINING WORDS EMBEDDING")
            with open(emo2vec_file.format(hidden_size), 'rb') as f:
                emo = pickle.load(f, encoding='latin1')
            self.emo2vec.weight.data.copy_(torch.from_numpy(emo))
            self.emo2vec.weight.requires_grad = self.allow_tuning_emb
        
        if self.glove_embedding:            
            self.embed = nn.Embedding(len(input_size), hidden_size, padding_idx=len(input_size) - 1) 
            self.embed.weight.data.copy_(torch.from_numpy(np.array(gen_embeddings(input_size, hidden_size)) ))
            self.embed.weight.requires_grad = self.allow_tuning_emb

        self.lstm = nn.LSTM(hidden_size*2, hidden_size*2, n_layers, bidirectional=bidirectional, batch_first=True,dropout=dr)        
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dr)

    def forward(self, input_emo, input_glove):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        batch_size = input.size(0)
        hidden = self._init_hidden(batch_size)
        emo2vec = self.dropout(self.emo2vec(input_emo))
        glove = self.dropout(self.embed(input_glove))
        _, hidden = self.lstm(torch.cat([emo2vec,glove],2), hidden)
        fc_output = self.fc(hidden[0][-1]) ## hidden[0] cause we take the hdd not the state
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size*2)
        state = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size*2)
        return (create_variable(hidden), create_variable(state))
    
    def trainer(self,dataset,opt,criterion,name):
        measure_avg = 0
        loss_avg = 0
        pbar = tqdm(enumerate(dataset),total=len(dataset))
        for i,d in pbar:
            opt.zero_grad() 
            out = self.forward(create_variable(d[0]), create_variable(d[1]))
            loss = criterion(out,create_variable(d[2]))
            measure = self.get_measure(name,out,d[2])# (out.max(dim=1)[1] == create_variable(d[2])).sum().data[0] / float(d[2].size(0))
            loss.backward()
            opt.step()
            measure_avg += measure
            loss_avg += loss.data[0]
            pbar.set_description("LOSS:{:.2f} MEAS:{:.2f}".format(loss_avg/float(i+1),measure_avg/float(i+1)))

    def eval(self,dataset,criterion,name):
        self.train(False)
        measure_avg = 0
        loss_avg = 0
        pbar = tqdm(enumerate(dataset),total=len(dataset))
        for i,d in pbar:
            out = self.forward(create_variable(d[0]), create_variable(d[1]))
            loss = criterion(out,create_variable(d[2]))
            measure = self.get_measure(name,out,d[2]) #(out.max(dim=1)[1] == create_variable(d[2])).sum().data[0] / float(d[2].size(0))
            measure_avg += measure
            loss_avg += loss.data[0]
            pbar.set_description("LOSS:{:.2f} MEAS:{:.2f}".format(loss_avg/float(i+1),measure_avg/float(i+1)))
        self.train(True)
        return measure_avg/float(len(dataset)),loss_avg/float(len(dataset))

    def get_measure(self,name,logit_batches,y_batch):
        # check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        if(dataset2class[name]==2):
            f1 = f1_score(y_batch.numpy(), logit_batches.max(dim=1)[1].data.cpu().numpy(), average='binary')
        else:
            f1 = f1_score(y_batch.numpy(), logit_batches.max(dim=1)[1].data.cpu().numpy(), average='macro')
        if (use_acc_dict[name]):
            measure = (logit_batches.max(dim=1)[1] == create_variable(y_batch)).sum().data[0] / float(y_batch.size(0))
        else:
            measure = f1

        return measure

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
