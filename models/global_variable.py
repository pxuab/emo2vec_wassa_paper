import os
import logging

extra_dim = 100

## EMO2VEC vocab
# vocab_path = "datasets/clean_hashtag/tokenized/vocab.pkl"
# emo2vec_file = "trained_models/emo2vec/emo2vec_dim_{}/emotion_embedding.pkl"
# emo2vec_model_file =  "trained_models/emo2vec/emo2vec_dim_{}/model_weights.npy"

vocab_path = "datasets/hashtag_full_vocab/idx/vocab.pkl"
emo2vec_file = "trained_models/emo2vec_full_vocab/dim_{}/emotion_embedding.pkl"
emo2vec_model_file =  "trained_models/emo2vec_full_vocab/dim_{}/model_weights.npy"
# emo2vec_file = "trained_models/emo2vec_full_w2v/dim_{}/emotion_embedding.pkl"
# emo2vec_model_file =  "trained_models/emo2vec_full_w2v/dim_{}/model_weights.npy"
dataset_names = ["tube_auto","abusive","cEXT_personality","cNEU_personality","cAGR_personality",
                "cCON_personality","cOPN_personality",
                "fear_SE0714","joy_SE0714","sad_SE0714",
                "valence_low_Olympic","valence_high_Olympic",
                "arousal_low_Olympic","arousal_high_Olympic",
                "stress","SCv1","SCv2",
                "kaggle","SS_T","SS_Y","isear",
                "wassa","semeval", "opener", "tube_auto",
                "tube_tablet", "sst_coarse", "sst_fine", "emo"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))



max_len = 66
if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False


dataset2class = {"abusive":2,"cEXT_personality":2,"cNEU_personality":2,"cAGR_personality":2,
                "cCON_personality":2,"cOPN_personality":2,"fear_SE0714" : 2,"joy_SE0714" : 2,
                "sad_SE0714" : 2,"valence_low_Olympic" : 2, "valence_high_Olympic" : 2, 
                "arousal_low_Olympic" : 2, "arousal_high_Olympic" : 2,"stress" : 2,
                "SCv1" : 2, "SCv2" : 2,"kaggle":2,"SS_T" : 2,"SS_Y" : 2,    
                "isear" : 7,"wassa" : 4,"semeval" : 3, "opener" : 4,
                "tube_auto" : 2, "tube_tablet" : 2,
                "sst_coarse" : 2, "sst_fine" : 5    
                 }


use_acc_dict = {"isear": False, "wassa": False, "SS_T": True, "SS_Y": True, "semeval": True, "opener":True, "tube_auto":True,
                 "tube_tablet":True, "sst_coarse":True, "sst_fine":True, "arousal_low_Olympic":False,"arousal_high_Olympic":False,"valence_low_Olympic": False,"valence_high_Olympic":False,
                "stress": False,"SCv1": False,"SCv2": False, "kaggle": False, "fear_SE0714":False,"joy_SE0714":False,"sad_SE0714":False,"cEXT_personality":False,"cNEU_personality":False,"cAGR_personality":False,
                "cCON_personality":False,"cOPN_personality": False, "isear_pos0": False,"isear_pos1": False, "isear_pos2": False, "isear_pos3": False, "isear_pos4": False, "isear_pos5": False,"isear_pos6":False, "joy_wassa":False, "sadness_wassa": False, "anger_wassa": False, "fear_wassa": False, "abusive": False} 

