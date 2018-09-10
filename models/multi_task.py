import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import datetime
import pickle
import os
import yaml
from .word_cnn import ConvModel
from models.global_variable import *
from utils.data_loader import gen_embeddings
import logging
from sklearn.metrics import f1_score

class config():

    def __init__(self, config_file):
        self.glove_file = None
        with open(config_file, 'r') as f:
            conf = yaml.load(f)
        for k, v in conf.items():
            setattr(self, k, v)


class MultitaskLearner(nn.Module):

    def __init__(self, vocab, vocab_task, embedding_dim, pretraind=False, dataset_names=None, dataset2class=None, extra_dim=None):
        super(MultitaskLearner, self).__init__()
        self.name = "MultitaskLearner"
        self.vocab_size = vocab
        self.vocab_task = vocab_task
        
        self.embedding_dim = embedding_dim
        self.emo2vec = nn.Embedding(
            vocab, embedding_dim, padding_idx=vocab - 1)

        self.dataset_names = dataset_names
        self.dataset2class = dataset2class

        self.use_combined_embedding = False
        self.extra_dim = extra_dim
            
        self.embed = nn.Embedding(
            len(vocab_task), self.extra_dim, padding_idx=1)   

        if (pretraind == 1):
            if(int(embedding_dim) == 100 or int(embedding_dim) == 200 or int(embedding_dim) == 300):
                with open(emo2vec_file.format(embedding_dim), 'rb') as f:
                    emo = pickle.load(f, encoding='latin1')
                # emo = np.vstack([emo, np.zeros(embedding_dim)])
                self.emo2vec.weight.data.copy_(torch.from_numpy(emo))
                self.emo2vec.weight.requires_grad = True
                logging.info("EMO2VEC LOADED")
            else:
                logging.info("WRONG EMBEDDING SIZE")

        if self.use_combined_embedding:
            emb_mat = np.array(gen_embeddings(vocab_task,self.extra_dim))
            self.embed.weight.data.copy_(torch.from_numpy(emb_mat))
            self.embed.weight.requires_grad = False

        # self.linear_models = {}
        # for d in self.dataset_names:
        #     if d == "emo":
        #         continue
        #     if self.use_combined_embedding:
        #         self.linear_models[d] = nn.Linear(2*embedding_dim, self.dataset2class[d]).cuda()
        #     else:
        #         self.linear_models[d] = nn.Linear(1*embedding_dim, self.dataset2class[d]).cuda()
        
        if self.use_combined_embedding:
            embedding_dim += self.extra_dim

        self.linear_semv = nn.Linear(embedding_dim, 3)  # semeval
        self.linear_SS_T = nn.Linear(embedding_dim, 2)  # SS_Twitter 
        self.linear_SS_Y = nn.Linear(embedding_dim, 2)  # SS_Youtube        
        self.linear_open = nn.Linear(embedding_dim, 4)  # opener
        self.linear_auto = nn.Linear(embedding_dim, 2)  # tube_auto
        self.linear_tabl = nn.Linear(embedding_dim, 2)  # tube_tablet
        self.liner_sst_c = nn.Linear(embedding_dim, 2)  # sst_coarse
        self.linear_sst_f = nn.Linear(embedding_dim, 5)  # sst_fine
        self.linear_wassa = nn.Linear(embedding_dim, 4)  # emotional
        self.linear_isear = nn.Linear(embedding_dim, 7)  # emotional
        
        self.linear_fear_SE0714 = nn.Linear(embedding_dim, 2)# emotional
        self.linear_joy_SE0714 = nn.Linear(embedding_dim, 2)# emotional
        self.linear_sad_SE0714 = nn.Linear(embedding_dim, 2)# emotional 
        self.linear_stress = nn.Linear(embedding_dim, 2) # stress
        self.linear_SCv1 = nn.Linear(embedding_dim, 2) # sarcasm
        self.linear_SCv2 = nn.Linear(embedding_dim, 2) # sarcasm
        self.linear_valence_low_Olympic = nn.Linear(embedding_dim, 2) # emotional
        self.linear_valence_high_Olympic = nn.Linear(embedding_dim, 2) # emotional
        self.linear_arousal_low_Olympic = nn.Linear(embedding_dim, 2) # emotional
        self.linear_arousal_high_Olympic = nn.Linear(embedding_dim, 2) # emotional

        self.linear_cEXT_personality = nn.Linear(embedding_dim, 2)# personality
        self.linear_cNEU_personality = nn.Linear(embedding_dim, 2)# personality
        self.linear_cAGR_personality = nn.Linear(embedding_dim, 2)# personality 
        self.linear_cCON_personality = nn.Linear(embedding_dim, 2)# personality 
        self.linear_cOPN_personality = nn.Linear(embedding_dim, 2)# personality 
        
        self.linear_kaggle = nn.Linear(embedding_dim, 2) # abusive
        self.linear_abusive = nn.Linear(embedding_dim, 2)# abusive        
        


        # self.cnn_model = nn.Linear(embedding_dim,4) ## EMO
        self.cnn_model = self.load_cnn_model()
        
        # self.cnn_model.cuda()

    def l2_regularization(self):
        l2_loss = 0
        if 'semeval' in self.dataset_names:
            l2_loss += self.linear_semv.weight.norm(2)
        if 'SS_T' in self.dataset_names:
            l2_loss += self.linear_SS_T.weight.norm(2)
        if 'SS_Y' in self.dataset_names:
            l2_loss += self.linear_SS_Y.weight.norm(2)
        if 'SS_Y' in self.dataset_names:
            l2_loss += self.linear_open.weight.norm(2)
        if 'tube_auto' in self.dataset_names:
            l2_loss += self.linear_auto.weight.norm(2)
        if 'tube_tablet' in self.dataset_names:
            l2_loss += self.linear_tabl.weight.norm(2)
        if 'sst_coarse' in self.dataset_names:
            l2_loss += self.liner_sst_c.weight.norm(2)
        if 'sst_fine' in self.dataset_names:
            l2_loss += self.linear_sst_f.weight.norm(2)
        if 'wassa' in self.dataset_names:
            l2_loss += self.linear_wassa.weight.norm(2)
        if 'isear' in self.dataset_names:
            l2_loss += self.linear_isear.weight.norm(2)

        if 'fear_SE0714' in self.dataset_names:
            l2_loss += self.linear_fear_SE0714.weight.norm(2)
        if 'joy_SE0714' in self.dataset_names:
            l2_loss += self.linear_joy_SE0714.weight.norm(2)
        if 'sad_SE0714' in self.dataset_names:
            l2_loss += self.linear_sad_SE0714.weight.norm(2)
        if 'stress' in self.dataset_names:
            l2_loss += self.linear_stress.weight.norm(2)
        if 'SCv1' in self.dataset_names:
            l2_loss += self.linear_SCv1.weight.norm(2)
        if 'SCv2' in self.dataset_names:
            l2_loss += self.linear_SCv2.weight.norm(2)
        if 'valence_low_Olympic' in self.dataset_names:
            l2_loss += self.linear_valence_low_Olympic.weight.norm(2)
        if 'valence_high_Olympic' in self.dataset_names:
            l2_loss += self.linear_valence_high_Olympic.weight.norm(2)
        if 'arousal_low_Olympic' in self.dataset_names:
            l2_loss += self.linear_arousal_low_Olympic.weight.norm(2)
        if 'arousal_high_Olympic' in self.dataset_names:
            l2_loss += self.linear_arousal_high_Olympic.weight.norm(2)
        if 'kaggle' in self.dataset_names:
            l2_loss += self.linear_kaggle.weight.norm(2)

        if 'cEXT_personality' in self.dataset_names:
            l2_loss += self.linear_cEXT_personality.weight.norm(2)
        if 'cNEU_personality' in self.dataset_names:
            l2_loss += self.linear_cNEU_personality.weight.norm(2)
        if 'cAGR_personality' in self.dataset_names:
            l2_loss += self.linear_cAGR_personality.weight.norm(2)
        if 'cCON_personality' in self.dataset_names:
            l2_loss += self.linear_cCON_personality.weight.norm(2)
        if 'cOPN_personality' in self.dataset_names:
            l2_loss += self.linear_cOPN_personality.weight.norm(2)
        if 'abusive' in self.dataset_names:
            l2_loss += self.linear_abusive.weight.norm(2)
        return l2_loss

    def save_model(self,model,acc):
        directory = 'save/{}/EMO_{}_WL_{}_L2_{}_LR_{}_BSZ_{}_EMB_SIZE_{}_ACC_{}/'.format(args['id'], args["emo2vec"],args['weightedloss'],args['l2'],args['lr'],args['bsz'],self.embedding_dim,acc)
        if args['remove_key']:
            assert args['id'] != 'n.a.'
            directory = 'save/{}/{}/EMO_{}_WL_{}_L2_{}_LR_{}_BSZ_{}_EMB_SIZE_{}_ACC_{}/'.format(args['id'], args['remove_key'], args["emo2vec"],args['weightedloss'],args['l2'],args['lr'],args['bsz'],self.embedding_dim,acc)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model, directory+'model.th')
        logging.info("MODEL SAVED")


    def load_cnn_model(self):
        dirname = os.path.dirname(__file__)
        yaml_file = os.path.join(dirname, "config/hashtag_cnn_100.yaml")
        FLAGS = config(yaml_file)

        filter_sizes = [int(f_z)
                        for f_z in FLAGS.filter_sizes.split(',') if f_z != '']
        
        if self.use_combined_embedding:
            embedding_size = self.embedding_dim + self.extra_dim
        else:
            embedding_size = self.embedding_dim

        cnn_model = ConvModel(max_len, 4, self.vocab_size,
                              filter_sizes, FLAGS.num_filters,
                              embedding_size=embedding_size,
                              dropout_prob=FLAGS.dropout)

        if not self.use_combined_embedding:
            cnn_model.load_model_from_ny(emo2vec_model_file.format(self.embedding_dim))
        return cnn_model

    def forward(self, task_batches, task_batches_emb):   

        # out_emo = self.cnn_model(self.emo2vec(task_batches["emo"]))
        # out = {"emo" : out_emo}
        # for d in self.dataset_names:
        #     if d == "emo":
        #         continue
        #     if self.use_combined_embedding:
        #         out[d] = self.linear_models[d](
        #           torch.sum(
        #           torch.cat([self.emo2vec(task_batches[d]),
        #           self.embed(task_batches_emb[d])],dim=2)
        #           , dim=1)) 
        #     else:
        #         out[d] = self.linear_models[d](
        #           torch.sum(
        #           self.emo2vec(task_batches[d])
        #           , dim=1))
   
        if self.use_combined_embedding:
            out_abusive = self.linear_abusive(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["abusive"]),
                    self.embed(task_batches_emb["abusive"])],dim=2)
                    , dim=1))
            out_cEXT_personality = self.linear_cEXT_personality(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["cEXT_personality"]),
                    self.embed(task_batches_emb["cEXT_personality"])],dim=2)
                    , dim=1))
            out_cNEU_personality = self.linear_cNEU_personality(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["cNEU_personality"]),
                    self.embed(task_batches_emb["cNEU_personality"])],dim=2)
                    , dim=1))
            out_cAGR_personality = self.linear_cAGR_personality(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["cAGR_personality"]),
                    self.embed(task_batches_emb["cAGR_personality"])],dim=2)
                    , dim=1))
            out_cCON_personality = self.linear_cCON_personality(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["cCON_personality"]),
                    self.embed(task_batches_emb["cCON_personality"])],dim=2)
                    , dim=1))
            out_cOPN_personality = self.linear_cOPN_personality(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["cOPN_personality"]),
                    self.embed(task_batches_emb["cOPN_personality"])],dim=2)
                    , dim=1))
            out_fear_SE0714 = self.linear_fear_SE0714(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["fear_SE0714"]),
                    self.embed(task_batches_emb["fear_SE0714"])],dim=2)
                    , dim=1))
            out_joy_SE0714 = self.linear_joy_SE0714(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["joy_SE0714"]),
                    self.embed(task_batches_emb["joy_SE0714"])],dim=2)
                    , dim=1))
            out_sad_SE0714 = self.linear_sad_SE0714(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["sad_SE0714"]),
                    self.embed(task_batches_emb["sad_SE0714"])],dim=2)
                    , dim=1))
            out_stress = self.linear_stress(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["stress"]),
                    self.embed(task_batches_emb["stress"])],dim=2)
                    , dim=1))
            out_SCv1 = self.linear_SCv1(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["SCv1"]),
                    self.embed(task_batches_emb["SCv1"])],dim=2)
                    , dim=1))
            out_SCv2 = self.linear_SCv2(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["SCv2"]),
                    self.embed(task_batches_emb["SCv2"])],dim=2)
                    , dim=1))
            out_valence_low_Olympic = self.linear_valence_low_Olympic(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["valence_low_Olympic"]),
                    self.embed(task_batches_emb["valence_low_Olympic"])],dim=2)
                    , dim=1))
            out_valence_high_Olympic = self.linear_valence_high_Olympic(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["valence_high_Olympic"]),
                    self.embed(task_batches_emb["valence_high_Olympic"])],dim=2)
                    , dim=1))
            out_arousal_low_Olympic = self.linear_arousal_low_Olympic(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["arousal_low_Olympic"]),
                    self.embed(task_batches_emb["arousal_low_Olympic"])],dim=2)
                    , dim=1))
            out_arousal_high_Olympic = self.linear_arousal_high_Olympic(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["arousal_high_Olympic"]),
                    self.embed(task_batches_emb["arousal_high_Olympic"])],dim=2)
                    , dim=1))
            out_kaggle = self.linear_kaggle(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["kaggle"]),
                    self.embed(task_batches_emb["kaggle"])],dim=2)
                    , dim=1))
            out_semv = self.linear_semv(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["semeval"]),
                    self.embed(task_batches_emb["semeval"])],dim=2)
                    , dim=1))
            out_wassa = self.linear_wassa(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["wassa"]),
                    self.embed(task_batches_emb["wassa"])],dim=2)
                    , dim=1))
            out_isear = self.linear_isear(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["isear"]),
                    self.embed(task_batches_emb["isear"])],dim=2)
                    , dim=1))
            out_SS_T = self.linear_SS_T(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["SS_T"]),
                    self.embed(task_batches_emb["SS_T"])],dim=2)
                    , dim=1))
            out_SS_Y = self.linear_SS_Y(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["SS_Y"]),
                    self.embed(task_batches_emb["SS_Y"])],dim=2)
                    , dim=1))
            out_open = self.linear_open(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["opener"]),
                    self.embed(task_batches_emb["opener"])],dim=2)
                    , dim=1))
            out_auto = self.linear_auto(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["tube_auto"]),
                    self.embed(task_batches_emb["tube_auto"])],dim=2)
                    , dim=1))
            out_tabl = self.linear_tabl(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["tube_tablet"]),
                    self.embed(task_batches_emb["tube_tablet"])],dim=2)
                    , dim=1))
            out_standC = self.liner_sst_c(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["sst_coarse"]),
                    self.embed(task_batches_emb["sst_coarse"])],dim=2)
                    , dim=1))
            out_standF = self.linear_sst_f(
                torch.sum(
                    torch.cat([self.emo2vec(task_batches["sst_fine"]),
                    self.embed(task_batches_emb["sst_fine"])],dim=2)
                    , dim=1))
            out_emo = self.cnn_model(
                    torch.cat([self.emo2vec(task_batches["emo"]),
                    self.embed(task_batches_emb["emo"])],dim=2))
                    
        else: 
            out_abusive = self.linear_abusive(
                torch.sum(
                    self.emo2vec(task_batches["abusive"])
                    , dim=1)) if 'abusive' in self.dataset_names else None
            out_cEXT_personality = self.linear_cEXT_personality(
                torch.sum(
                    self.emo2vec(task_batches["cEXT_personality"])
                    , dim=1)) if 'cEXT_personality' in self.dataset_names else None
            out_cNEU_personality = self.linear_cNEU_personality(
                torch.sum(
                    self.emo2vec(task_batches["cNEU_personality"])
                    , dim=1)) if 'cNEU_personality' in self.dataset_names else None
            out_cAGR_personality = self.linear_cAGR_personality(
                torch.sum(
                    self.emo2vec(task_batches["cAGR_personality"])
                    , dim=1)) if 'cAGR_personality' in self.dataset_names else None
            out_cCON_personality = self.linear_cCON_personality(
                torch.sum(
                    self.emo2vec(task_batches["cCON_personality"])
                    , dim=1)) if 'cCON_personality' in self.dataset_names else None
            out_cOPN_personality = self.linear_cOPN_personality(
                torch.sum(
                    self.emo2vec(task_batches["cOPN_personality"])
                    , dim=1)) if 'cOPN_personality' in self.dataset_names else None
            out_fear_SE0714 = self.linear_fear_SE0714(
                torch.sum(
                    self.emo2vec(task_batches["fear_SE0714"])
                    , dim=1)) if 'fear_SE0714' in self.dataset_names else None
            out_joy_SE0714 = self.linear_joy_SE0714(
                torch.sum(
                    self.emo2vec(task_batches["joy_SE0714"])
                    , dim=1)) if 'joy_SE0714' in self.dataset_names else None
            out_sad_SE0714 = self.linear_sad_SE0714(
                torch.sum(
                    self.emo2vec(task_batches["sad_SE0714"])
                    , dim=1)) if 'sad_SE0714' in self.dataset_names else None
            out_stress = self.linear_stress(
                torch.sum(
                    self.emo2vec(task_batches["stress"])
                    , dim=1)) if 'stress' in self.dataset_names else None
            out_SCv1 = self.linear_SCv1(
                torch.sum(
                    self.emo2vec(task_batches["SCv1"])
                    , dim=1)) if 'SCv1' in self.dataset_names else None
            out_SCv2 = self.linear_SCv2(
                torch.sum(
                    self.emo2vec(task_batches["SCv2"])
                    , dim=1)) if 'SCv2' in self.dataset_names else None
            out_valence_low_Olympic = self.linear_valence_low_Olympic(
                torch.sum(
                    self.emo2vec(task_batches["valence_low_Olympic"])
                    , dim=1)) if 'valence_low_Olympic' in self.dataset_names else None
            out_valence_high_Olympic = self.linear_valence_high_Olympic(
                torch.sum(
                    self.emo2vec(task_batches["valence_high_Olympic"])
                    , dim=1)) if 'valence_high_Olympic' in self.dataset_names else None
            out_arousal_low_Olympic = self.linear_arousal_low_Olympic(
                torch.sum(
                    self.emo2vec(task_batches["arousal_low_Olympic"])
                    , dim=1)) if 'arousal_low_Olympic' in self.dataset_names else None
            out_arousal_high_Olympic = self.linear_arousal_high_Olympic(
                torch.sum(
                    self.emo2vec(task_batches["arousal_high_Olympic"])
                    , dim=1)) if 'arousal_high_Olympic' in self.dataset_names else None
            out_kaggle = self.linear_kaggle(
                torch.sum(
                    self.emo2vec(task_batches["kaggle"])
                    , dim=1)) if 'kaggle' in self.dataset_names else None
            out_semv = self.linear_semv(
                torch.sum(
                    self.emo2vec(task_batches["semeval"])
                    , dim=1)) if 'semeval' in self.dataset_names else None
            out_wassa = self.linear_wassa(
                torch.sum(
                    self.emo2vec(task_batches["wassa"])
                    , dim=1)) if 'wassa' in self.dataset_names else None
            out_isear = self.linear_isear(
                torch.sum(
                    self.emo2vec(task_batches["isear"])
                    , dim=1)) if 'isear' in self.dataset_names else None
            out_SS_T = self.linear_SS_T(
                torch.sum(
                    self.emo2vec(task_batches["SS_T"])
                    , dim=1)) if 'SS_T' in self.dataset_names else None
            out_SS_Y = self.linear_SS_Y(
                torch.sum(
                    self.emo2vec(task_batches["SS_Y"])
                    , dim=1)) if 'SS_Y' in self.dataset_names else None
            out_open = self.linear_open(
                torch.sum(
                    self.emo2vec(task_batches["opener"])
                    , dim=1)) if 'opener' in self.dataset_names else None
            out_auto = self.linear_auto(
                torch.sum(
                    self.emo2vec(task_batches["tube_auto"])
                    , dim=1)) if 'tube_auto' in self.dataset_names else None
            out_tabl = self.linear_tabl(
                torch.sum(
                    self.emo2vec(task_batches["tube_tablet"])
                    , dim=1)) if 'tube_tablet' in self.dataset_names else None
            out_standC = self.liner_sst_c(
                torch.sum(
                    self.emo2vec(task_batches["sst_coarse"])
                    , dim=1)) if 'sst_coarse' in self.dataset_names else None
            out_standF = self.linear_sst_f(
                torch.sum(
                    self.emo2vec(task_batches["sst_fine"])
                    , dim=1)) if 'sst_fine' in self.dataset_names else None

            out_emo = self.cnn_model(self.emo2vec(task_batches["emo"]))

        # return out 
        return {
                "abusive": out_abusive,
                "cEXT_personality": out_cEXT_personality,
                "cNEU_personality": out_cNEU_personality,
                "cAGR_personality": out_cAGR_personality,
                "cCON_personality": out_cCON_personality,
                "cOPN_personality": out_cOPN_personality,
                "fear_SE0714": out_fear_SE0714,
                "joy_SE0714": out_joy_SE0714,
                "sad_SE0714": out_sad_SE0714,
                "stress": out_stress,
                "SCv1": out_SCv1,
                "SCv2": out_SCv2,
                "valence_low_Olympic": out_valence_low_Olympic,
                "valence_high_Olympic": out_valence_high_Olympic,
                "arousal_low_Olympic": out_arousal_low_Olympic,
                "arousal_high_Olympic": out_arousal_high_Olympic,
                "kaggle": out_kaggle,
                "emo": out_emo,
                "semeval": out_semv,
                "wassa": out_wassa,
                "isear": out_isear,
                "SS_T": out_SS_T,
                "SS_Y": out_SS_Y,
                "opener": out_open, 
                "tube_auto": out_auto,
                "tube_tablet": out_tabl,
                "sst_coarse": out_standC,
                "sst_fine": out_standF}

    def predict_single(self, inp,inp_emb, data_set):

        # if data_set != "emo":
        #     if self.use_combined_embedding:
        #         out = self.linear_models[data_set](torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
        #     else:
        #         out = self.linear_models[data_set](torch.sum(self.emo2vec(inp), dim=1))
        # else:
        #     if self.use_combined_embedding:
        #         out = self.cnn_model(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
        #     else:
        #         out = self.cnn_model(torch.sum(self.emo2vec(inp), dim=1))   
        if self.use_combined_embedding:
            if(data_set == 'abusive'):
                out = self.linear_abusive(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'cEXT_personality'):
                out = self.linear_cEXT_personality(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'cNEU_personality'):
                out = self.linear_cNEU_personality(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'cAGR_personality'):
                out = self.linear_cAGR_personality(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'cCON_personality'):
                out = self.linear_cCON_personality(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'cOPN_personality'):
                out = self.linear_cOPN_personality(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'fear_SE0714'):
                out = self.linear_fear_SE0714(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'joy_SE0714'):
                out = self.linear_joy_SE0714(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'sad_SE0714'):
                out = self.linear_sad_SE0714(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'stress'):
                out = self.linear_stress(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'SCv1'):
                out = self.linear_SCv1(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'SCv2'):
                out = self.linear_SCv2(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'valence_low_Olympic'):
                out = self.linear_valence_low_Olympic(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'valence_high_Olympic'):
                out = self.linear_valence_high_Olympic(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'arousal_low_Olympic'):
                out = self.linear_arousal_low_Olympic(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'arousal_high_Olympic'):
                out = self.linear_arousal_high_Olympic(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'kaggle'):
                out = self.linear_kaggle(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'semeval'):
                out = self.linear_semv(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'wassa'):
                out = self.linear_wassa(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'isear'):
                out = self.linear_isear(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'SS_T'):
                out = self.linear_SS_T(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            if(data_set == 'SS_Y'):
                out = self.linear_SS_Y(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            elif(data_set == 'opener'):
                out = self.linear_open(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            elif(data_set == 'tube_auto'):
                out = self.linear_auto(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            elif(data_set == 'tube_tablet'):
                out = self.linear_tabl(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            elif(data_set == 'sst_coarse'):
                out = self.liner_sst_c(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            elif(data_set == 'sst_fine'):
                out = self.linear_sst_f(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
            elif(data_set == 'emo'):
                out = self.cnn_model(torch.sum(torch.cat([self.embed(inp_emb),self.emo2vec(inp)],dim=2), dim=1))
        else:
            if(data_set == 'abusive'):
                out = self.linear_abusive(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'cEXT_personality'):
                out = self.linear_cEXT_personality(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'cNEU_personality'):
                out = self.linear_cNEU_personality(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'cAGR_personality'):
                out = self.linear_cAGR_personality(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'cCON_personality'):
                out = self.linear_cCON_personality(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'cOPN_personality'):
                out = self.linear_cOPN_personality(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'fear_SE0714'):
                out = self.linear_fear_SE0714(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'joy_SE0714'):
                out = self.linear_joy_SE0714(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'sad_SE0714'):
                out = self.linear_sad_SE0714(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'stress'):
                out = self.linear_stress(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'SCv1'):
                out = self.linear_SCv1(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'SCv2'):
                out = self.linear_SCv2(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'valence_low_Olympic'):
                out = self.linear_valence_low_Olympic(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'valence_high_Olympic'):
                out = self.linear_valence_high_Olympic(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'arousal_low_Olympic'):
                out = self.linear_arousal_low_Olympic(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'arousal_high_Olympic'):
                out = self.linear_arousal_high_Olympic(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'kaggle'):
                out = self.linear_kaggle(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'semeval'):
                out = self.linear_semv(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'wassa'):
                out = self.linear_wassa(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'isear'):
                out = self.linear_isear(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'SS_T'):
                out = self.linear_SS_T(torch.sum(self.emo2vec(inp), dim=1))
            if(data_set == 'SS_Y'):
                out = self.linear_SS_Y(torch.sum(self.emo2vec(inp), dim=1))
            elif(data_set == 'opener'):
                out = self.linear_open(torch.sum(self.emo2vec(inp), dim=1))
            elif(data_set == 'tube_auto'):
                out = self.linear_auto(torch.sum(self.emo2vec(inp), dim=1))
            elif(data_set == 'tube_tablet'):
                out = self.linear_tabl(torch.sum(self.emo2vec(inp), dim=1))
            elif(data_set == 'sst_coarse'):
                out = self.liner_sst_c(torch.sum(self.emo2vec(inp), dim=1))
            elif(data_set == 'sst_fine'):
                out = self.linear_sst_f(torch.sum(self.emo2vec(inp), dim=1))
            elif(data_set == 'emo'):
                out = self.cnn_model(torch.sum(self.emo2vec(inp), dim=1))
        return out


class Trainer():

    def __init__(self, opt, loss, dataset_names):
        self.step = 0
        self.loss = 0
        self.acc = 0
        self.opt = opt
        self.H = loss
        self.dataset_names = dataset_names
        self.details = {n: [0, 0] for n in self.dataset_names}

    def train(self, X_batch, X_batchesEMB, y_batch, model, weight_loss, sigma=0.1):
        weight_loss.append(1)
        self.opt.zero_grad()
        logit_batches = model(X_batch,X_batchesEMB)
        loss_total = 0
        acc_total = 0
        for i, n in enumerate(self.dataset_names):
            # if n == "emo":
            #     continue
            if USE_CUDA:
                temp_loss = self.H(logit_batches[n], y_batch[
                                   n]) * Variable(torch.Tensor([weight_loss[i]]))[0].cuda()
            else:
                temp_loss = self.H(logit_batches[n], y_batch[
                                   n]) * Variable(torch.Tensor([weight_loss[i]]))[0]

            loss_total += temp_loss
            self.details[n][0] += temp_loss.data[0]

            temp_acc = (logit_batches[n].max(dim=1)[1] == y_batch[
                        n]).sum().data[0] / float(y_batch[n].size(0))
            acc_total += temp_acc
            self.details[n][1] += temp_acc

        loss_total += sigma * model.l2_regularization()
        #  backprop
        loss_total.backward()
        self.opt.step()
        self.step += 1
        self.loss += loss_total.data[0]
        self.acc += acc_total / len(self.dataset_names)

    def predict(self, dataset_names, data, model):
        logging.info("Evaluate")
        val_acc = {}
        for n in dataset_names:
            acc = 0
            for i, (X, X_EMB,y) in enumerate(data[n]):
                if(USE_CUDA):
                    acc += self.pred_batch(Variable(X).cuda(),Variable(X_EMB).cuda(), 
                                           Variable(y).cuda(), model, n)
                else:
                    acc += self.pred_batch(Variable(X), Variable(X_EMB), Variable(y), model, n)
            acc = acc / float(len(data[n]))
            logging.info("Acc:{:.2f} {}".format(acc, n))
            val_acc[n] = acc
        return val_acc

    def pred_batch(self, X_batch,X_EMB, y_batch, model, name):
        model.train(False)
        logit_batches = model.predict_single(X_batch,X_EMB, name)
        acc = (logit_batches.max(dim=1)[1] == y_batch).sum().data[0] / float(y_batch.size(0))
        ## check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        # if(dataset2class[name]==2):
        #     f1 = f1_score(y_batch.data.cpu().numpy(), logit_batches.max(dim=1)[1].data.cpu().numpy(), average='binary')
        # else:
        #     f1 = f1_score(y_batch.data.cpu().numpy(), logit_batches.max(dim=1)[1].data.cpu().numpy(), average='macro')
        # if (use_acc_dict[name]):
        #     pass
        # else:
        #     acc = f1
            
        model.train(True)
        return acc

    def print_loss(self, e):
        logging.info("Epoch:{}".format(e))
        logging.info("TOTAL Loss:{:.2f} Acc:{:.2f}".format(
            self.loss / float(self.step), self.acc / float(self.step)))
        for n in self.dataset_names:
            logging.info("Loss:{:.2f} Acc:{:.2f} {} ".format(self.details[n][
                  0] / float(self.step), self.details[n][1] / float(self.step), n))
            self.details[n][0] = 0
            self.details[n][1] = 0
        self.step = 0
        self.loss = 0
        self.acc = 0
