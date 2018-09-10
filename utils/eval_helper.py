import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from models.global_variable import use_acc_dict
from torch import nn
import torch

# labels = {"anger":0, "fear":1, "joy":2, "sadness":3}
# labels_reverse = {0:"anger", 1:"fear", 2:"joy", 3:"sadness"}

def find_f1_threshold(y_val, y_pred_val, y_test, y_pred_test,
                      average='binary'):
    """ Choose a threshold for F1 based on the validation dataset
        (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/
        for details on why to find another threshold than simply 0.5)

    # Arguments:
        y_val: Outputs of the validation dataset.
        y_pred_val: Predicted outputs of the validation dataset.
        y_test: Outputs of the testing dataset.
        y_pred_test: Predicted outputs of the testing dataset.

    # Returns:
        F1 score for the given data and
        the corresponding F1 threshold
    """
    if y_pred_val is None:
        best_t = 0.5
    else:
        thresholds = np.arange(0.01, 0.5, step=0.01)
        f1_scores = []

        for t in thresholds:
            y_pred_val_ind = (y_pred_val > t)
            f1_val = f1_score(y_val, y_pred_val_ind, average=average)
            f1_scores.append(f1_val)

        best_t = thresholds[np.argmax(f1_scores)]
       
    y_pred_ind = (y_pred_test > best_t)
    f1_test = f1_score(y_test, y_pred_ind, average=average)
    return y_pred_ind,  f1_test

def fit_and_predict_sklearn(x_train, x_test, y_train, y_test, y_valid, y_pred_valid, clf, labels=None,
                            return_preds=False, return_clf=False, use_acc=True,
                            reverse=False):
    if reverse:
        x_train, x_test = x_test, x_train
        y_test, y_train = y_train, y_test

    print("x_train:%s, x_test:%s" % (x_train.shape, x_test.shape))
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    preds_prob = clf.predict_proba(x_test)[:,1]
    extras = {}
    if return_preds:
        extras["preds"] = preds
        extras["probs"] = clf.predict_proba(x_test)
    if return_clf:
        extras["clf"] = clf

    if len(extras) == 0:
        if use_acc:
            return (accuracy_score(y_test, preds),
                    classification_report(y_test, preds, digits=3,
                                      target_names=labels))
        else: 
            extras["preds"], best_f1 = find_f1_threshold(y_valid, y_pred_valid, y_test, preds_prob)
            return (best_f1,
                    classification_report(y_test, preds, digits=3,
                                      target_names=labels))
    else:
        if use_acc:
            return (accuracy_score(y_test, preds),
                    classification_report(y_test, preds, digits=3,
                    target_names=labels), extras)
        else: 
            extras["preds"], best_f1 = find_f1_threshold(y_valid, y_pred_valid, y_test, preds_prob)
            return (best_f1,
                    classification_report(y_test, preds, digits=3,
                    target_names=labels), extras)

def fit_and_predict_svm(x_train, x_test, y_train, y_test, labels=None,
                        return_preds=False, return_clf=False,
                        reverse=False):
    x_train, x_test = concat_features_if_needed(x_train, x_test)
    return fit_and_predict_sklearn(x_train, x_test, y_train, y_test,
                                   svm.LinearSVC(), labels=labels,
                                   return_clf=return_clf,
                                   return_preds=return_preds,
                                   reverse=reverse)

def fit_and_predict_lr(x_train, x_valid, x_test, y_train, y_valid, y_test, binary, use_acc=True, labels=None,
                       return_preds=False, return_clf=False, reverse=False, verbose=True):

    best_dim = 0
    best_C = None
    best_metric = 0
    best_pred = None

    for dim in x_train.keys():
        C, pred, metric = fit_and_validate_lr(x_train[dim], x_valid[dim], y_train, y_valid, binary=binary, use_acc=use_acc, labels=labels, verbose=verbose)
        if metric >= best_metric:
            best_C = C
            best_dim = dim
            best_metric = metric
            best_pred = pred
    if verbose:
       print("best dim is {}, best C is {}, best metric is {}".format(best_dim,best_C,best_metric))
    if binary:
        return fit_and_predict_sklearn(np.vstack((x_train[best_dim], x_valid[best_dim])), x_test[best_dim],
                                   np.concatenate((y_train, y_valid)),  y_test, y_valid, best_pred,
                                   LogisticRegression(C=best_C),
                                   return_clf=return_clf, use_acc=use_acc,
                                   return_preds=return_preds,
                                   reverse=reverse, labels=labels)
    else:
        return fit_and_predict_sklearn(np.vstack((x_train[best_dim], x_valid[best_dim])), x_test[best_dim],
                                   np.concatenate((y_train, y_valid)),  y_test, y_valid, best_pred,
                                   LogisticRegression(C=best_C, multi_class='multinomial', solver='newton-cg'),
                                   return_clf=return_clf, use_acc=use_acc,
                                   return_preds=return_preds,
                                   reverse=reverse, labels=labels)

def fit_and_validate_lr(x_train, x_valid, y_train, y_valid, binary, use_acc=True, labels=None, verbose=True):

    # Cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    Cs =    [0.0001, 0.001, 0.003, 0.006, 0.009,
                   0.01,  0.03,  0.06,  0.09,
                   0.1,   0.3,   0.6,   0.9,
                   1.,       3.,    6.,     9.,
                   10.,      30.,   60.,    90., 1000., 10000.0]
    best_pred = None
    best_metric, best_C = 0.0, Cs[-1]
    for Ci in Cs:
        if binary:
            lr = LogisticRegression(C=Ci)
        else:
            lr = LogisticRegression(C=Ci, multi_class='multinomial', solver='newton-cg')
        lr.fit(x_train, y_train)
        pred_valid = lr.predict(x_valid)
        if use_acc:
            acc = accuracy_score(y_valid, pred_valid)
            if acc > best_metric:
                best_metric, best_C = acc, Ci
                best_pred = None
            if verbose:
                print(acc, Ci)
        else:
            f1 = f1_score(y_valid, pred_valid, average="binary")
            if f1 > best_metric:
                best_metric, best_C = f1, Ci
                best_pred = lr.predict_proba(x_valid)[:,1]
            if verbose:
                print(f1, Ci)
    print("best c is ",best_C)
    if not use_acc and  best_pred is None:
        best_pred = lr.predict_proba(x_valid)[:,1]

    return best_C, best_pred, best_metric

def gen_rep(dataset, embeddings, word2id, emb_type):

    emb_table = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
    emb_table.weight.data.copy_(torch.from_numpy(embeddings))
    emb_table.weight.requires_grad = False

    padding_idx = word2id['PAD']
    # print("padding emb is ", embeddings[padding_idx])
    if emb_type == "glove":
        res = emb_table(dataset.X_dataEMB).sum(dim=1).data.numpy() / torch.clamp((dataset.X_dataEMB != padding_idx).sum(dim=1,keepdim=True), min=1).float()
        # res[res == float("Inf")] = 0
        return res
    elif emb_type == "emotion":
        res = emb_table(dataset.X_data).sum(dim=1).data.numpy() / torch.clamp((dataset.X_data != padding_idx).sum(dim=1,keepdim=True), min=1).float()
        # res[res == float("Inf")] = 0
        return res
    else:
        raise ValueError("invalid emb_type")

def process_result(target):
    tmp = [target[k] for k in target.keys() if "Olympic" in k]
    if len(tmp) > 0:
        target["Olympic"] = sum(tmp) / len(tmp)
    # print("Average of Olympic is",sum(tmp) / len(tmp))
    tmp = [target[k] for k in target.keys() if "SE0714" in k]
    if len(tmp) > 0:
        target["SE0714"] = sum(tmp) / len(tmp)
    # print("Average of SE0714 is",sum(tmp) / len(tmp))
    tmp = [target[k] for k in target.keys() if "personality" in k]
    if len(tmp) > 0:
        target["personality"] = sum(tmp) / len(tmp)
    # print("Average of personality is",sum(tmp) / len(tmp))
    tmp = [target[k] for k in target.keys() if "isear" in k]
    if len(tmp) > 0:
        target["isear"] = sum(tmp) / len(tmp)
    tmp = [target[k] for k in target.keys() if "wassa" in k]
    if len(tmp) > 0:
        target["wassa"] = sum(tmp) / len(tmp)
    return target

def train_predict_lr(dataset, embeddings, word2id, emb_type, d_name, return_preds=False, return_clf=False):

    print('-'*100)
    print('d_name: ', d_name)
    rep = {}
    label = {}
    for type_d in ['train', 'dev', 'test']:
        label[type_d] = np.array(dataset[type_d].y_data)
        rep[type_d] = {}
        for dim, embedding  in embeddings.items():
            rep[type_d][dim] = gen_rep(dataset[type_d], embedding, word2id, emb_type)

    if return_preds or return_clf:
        score, report, extra = fit_and_predict_lr(rep["train"], rep["dev"],rep["test"],
                                                       label["train"], label["dev"], label["test"], use_acc=use_acc_dict[d_name], binary=True,
                                                       return_preds=return_preds, return_clf=return_clf)
        return score, extra
    else:
        score, report = fit_and_predict_lr(rep["train"], rep["dev"],rep["test"],
                                                       label["train"], label["dev"], label["test"], use_acc=use_acc_dict[d_name], binary=True)
        return score

def train_predict_lr_combined(dataset, word_embeddings, vocab, emotion_embeddings, emotion_vocab, d_name, return_preds=False, return_clf=False):

    print('-'*100)
    print('d_name: ', d_name)
    rep = {}
    label = {}
    for type_d in ['train', 'dev', 'test']:
        label[type_d] = np.array(dataset[type_d].y_data)
        rep[type_d] = {}
        for dim, embeddings  in word_embeddings.items():
            rep[type_d][dim] = np.concatenate((gen_rep(dataset[type_d], embeddings, vocab, emb_type="glove"), gen_rep(dataset[type_d], emotion_embeddings, emotion_vocab, emb_type="emotion")), axis=1)

    if return_preds or  return_clf:
        score, report, extra = fit_and_predict_lr(rep["train"], rep["dev"],rep["test"],
                                                   label["train"], label["dev"], label["test"], use_acc=use_acc_dict[d_name], binary=True, 
                                                   return_preds=return_preds, return_clf=return_clf) 
        return score, extra
    else:
        score, report = fit_and_predict_lr(rep["train"], rep["dev"],rep["test"],
                                                       label["train"], label["dev"], label["test"], use_acc=use_acc_dict[d_name], binary=True)
        return  score

def evaluate_single_emb(datasets, embeddings, word2id, emb_type, name):

    results = {}
    for d_name, dataset in datasets.items():
        results[d_name] = train_predict_lr(dataset, embeddings, word2id, emb_type, d_name)

    results = process_result(results)
    results["type"] = name
    return results

def evaluate_combined_emb(datasets, word_embeddings, vocab, emotion_embeddings, emotion_vocab, name):

    results = {}
    for d_name, dataset in datasets.items():
        results[d_name] = train_predict_lr_combined(dataset, word_embeddings, vocab, emotion_embeddings, emotion_vocab, d_name)

    results = process_result(results)
    results["type"] = name
    return results

def save_to_csv(output_file, content, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["type", "SS_T","SS_Y", "sst_coarse", "sst_fine", "opener", "tube_auto","tube_tablet", "semeval", "isear", "wassa", "SE0714", "Olympic", "stress", "SCv1","SCv2", "kaggle", "abusive","personality"]
    import csv
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in content:
            valid_row = {key: value for key, value in i.items() if key in fieldnames}
            writer.writerow(valid_row)
    print("finish writing")
