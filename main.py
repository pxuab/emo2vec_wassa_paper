from utils.data_loader import get_data, get_batch
import torch.nn as nn
import torch
from torch import optim
from models.multi_task import MultitaskLearner, Trainer
from models.global_variable import *
import logging
import pickle
from models.arguments import args

if args['path']:
    path = args['path']
    if USE_CUDA:
        logging.info("MODEL {} LOADED".format(str(path)))
        model = torch.load(str(path)+'/model.th')
    else:
        logging.info("MODEL {} LOADED".format(str(path)))
        model = torch.load(str(path)+'/model.th',lambda storage, loc: storage)
    with open(path+"/emotion_embedding.pkl", 'wb') as f:
        pickle.dump(model.emo2vec.weight.data.cpu().numpy(), f)

    if not args['eval']:
        import sys
        sys.exit(0)

dataset_names = [d_name for d_name in dataset_names if args['remove_key'] not in d_name]
logging.info("valid datasets {}".format(dataset_names))

data_loader_TRAIN, data_loader_DEV, data_loader_TEST, vocab_len, vocab_task = get_data(
    dataset_names, bsz=int(args['bsz']), extra_dim=extra_dim)


if args['path']:
    if args['eval']:
        opt = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=float(args['lr']))
        trainer = Trainer(opt, nn.CrossEntropyLoss(), dataset_names)
        logging.info("VALIDATION ACC")
        acc_val = trainer.predict(dataset_names[:-1], data_loader_DEV, model)
        logging.info("TEST ACC")
        acc_val = trainer.predict(dataset_names[:-1], data_loader_TEST, model)

else:
    model = MultitaskLearner(vocab_len, vocab_task, int(args['esize']), pretraind=int(args["emo2vec"]), dataset_names=dataset_names, dataset2class=dataset2class, extra_dim=extra_dim)
    if USE_CUDA:
        logging.info("USE_CUDA TRUE")
        model.cuda()
    opt = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=float(args['lr']))
    trainer = Trainer(opt, nn.CrossEntropyLoss(), dataset_names)
    weight_loss = [1 for i in range(len(dataset_names) - 1)]
    best_avg = 0.0
    for e in range(10000000):
        X_batches, X_batchesEMB, y_batches = get_batch(dataset_names, data_loader_TRAIN)
        trainer.train(X_batches, X_batchesEMB, y_batches, model, weight_loss, sigma=float(args['l2']))
        if(e % 20 == 0):
            trainer.print_loss(e)
            acc_val = trainer.predict(dataset_names[:-1], data_loader_DEV, model)
            if (int(args['weightedloss'])==1):
                weight_loss = [1 - v for k, v in acc_val.items()]
            acc = sum([v for k, v in acc_val.items()]) / len(acc_val)
            logging.info("AVG acc: {:.2f}".format(acc))
            logging.info("")
            if (acc >= best_avg):
                best_avg = acc
                cnt = 0
                model.save_model(model,acc)
            else:
                cnt += 1
            if(cnt >= 100):
                break
