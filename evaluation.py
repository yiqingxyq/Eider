import os
import os.path
import json
import numpy as np

import pickle
from collections import defaultdict
from difflib import get_close_matches
from IPython import embed
from tqdm import tqdm

dataset_path = 'dataset/'
rel2id = json.load(open(dataset_path + 'meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}

def cal_f1(prec, recall):
    return (2*prec*recall)/(prec+recall)

def token2eid(pred, named_entities):
    pred = pred.lower()
    nes = ['none']
    ne2eid = {'none':-1}
    for i,e in enumerate(named_entities):
        nes.extend(e)
        for m in e:
            ne2eid[m] = i
    match = get_close_matches(pred, nes)[0]
    match_id = ne2eid[match]

    return match_id

def gen_official(tokens, features):
    res = []
    for i in range(preds.shape[0]):
        pred_t = token2eid(preds[i], features[i]['named_entities'])
        if pred_t != -1:
            h,_,r = features[i]['htr']
            res.append(
                {
                    'title': features[i]['title'],
                    'h_idx': h,
                    't_idx': pred_t,
                    'r': rs
                }
            )
    return res

def to_score(scores, topks, features):
    h_idx, t_idx, title = [], [], []
    for f in features:
        if 'original_hts' in f:
            hts = f['original_hts']
        else:
            hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]


    title2scores = {}
    k = topks.shape[-1]
    for i in range(len(scores)):
        tit, h, t = title[i], h_idx[i], t_idx[i]

        score_label = [(scores[i][j], topks[i][j]) for j in range(k)]

        ht_key = (h,t)
        if tit not in title2scores:
            title2scores[tit] = {}
        if ht_key not in title2scores[tit]:
            title2scores[tit][ht_key] = []
        title2scores[tit][ht_key].append(score_label)

    return title2scores

def extract_relative_score(scores):
    # scores: [score_i, ...]; where score_i: [(score_ik, rel_ik), ...]
    rel2rel_score = defaultdict(lambda: -100)
    for score in scores:
        na_score = score[-1][0] - 1
        for s, rel in score:
            if rel == 0:
                na_score = s
        for s, rel in score:
            if rel != 0:
                if rel in rel2rel_score:
                    rel2rel_score[rel] = max(rel2rel_score[rel], s-na_score)
                else:
                    rel2rel_score[rel] = s-na_score
    return rel2rel_score

def extract_gt(feature_path, features):
    gt_file = feature_file = os.path.join(feature_path, 'title2gt.pkl' )

    if os.path.exists(gt_file):
        title2gt = pickle.load(open(gt_file, 'rb'))
    else:
        print('Extracting gt..')
        title2gt = {}
        for f in tqdm(features):
            title = f['title']
            title2gt[title] = {}
            for idx,p in enumerate(f['hts']):
                h,t = p
                label = np.array(f['labels'][idx])
                rs = np.nonzero(label[1:])[0] + 1
                title2gt[title][(h,t)] = rs

        print('Saving title2gt to file..')
        pickle.dump(title2gt, open(gt_file, 'wb'))

    return title2gt

def ensemble_scores(title2scores, title2scores2, title2gt=None, thresh=None):

    assert(title2gt is not None or thresh is not None)

    res = []
    thresh_r_scores= []
    num_fixed_correct = 0
    num_fixed_pred = 0
    num_gt = 0

    num_pred2 = 0

    for title in title2scores:
        if title2gt is not None:
            gt = title2gt[title]
        ps = set(title2scores[title].keys())
        for h,t in ps:
            if title2gt is not None:
                num_gt += len(gt[(h,t)])
            rel2rel_score1 = extract_relative_score(title2scores[title][(h,t)])

            if title not in title2scores2 or (h,t) not in title2scores2[title]:
                tmp_res = [rel for rel in rel2rel_score1 if rel2rel_score1[rel] > 0]
            else:
                rel2rel_score2 = extract_relative_score(title2scores2[title][(h,t)])
                num_pred2 += len([rel for rel in rel2rel_score2 if rel2rel_score2[rel] > 0])

                rels = set(rel2rel_score1.keys()).union(set(rel2rel_score2.keys()))
                rel2rel_score = {rel:rel2rel_score1[rel] + rel2rel_score2[rel] for rel in rels}

                if thresh is not None:
                    tmp_res = [rel for rel in rels if (rel2rel_score1[rel] > 0 or rel2rel_score2[rel] > 0) and rel2rel_score[rel] >= thresh]
                else:
                    tmp_res = []
                    for rel in rels:
                        if rel2rel_score1[rel] > 0 and rel2rel_score2[rel] > 0:
                            tmp_res.append(rel)
                        elif rel2rel_score1[rel] > 0 or rel2rel_score2[rel] > 0:
                            if_correct = rel in gt[(h,t)]
                            thresh_r_scores.append( (if_correct, rel2rel_score[rel], title, h, t, rel) )

            num_fixed_pred += len(tmp_res)
            for r in tmp_res:
                if title2gt is not None:
                    if r in gt[(h,t)]:
                        num_fixed_correct += 1

                tmp_dict = {
                    'title': title,
                    'h_idx': h,
                    't_idx': t,
                    'r': id2rel[r],
                }
                res.append(tmp_dict)

    if thresh is not None or len(thresh_r_scores) == 0:
        return res, thresh
    else:
        thresh = {}

    print('# fixed pred:', num_fixed_pred, '# fixed correct:', num_fixed_correct, '# gt:', num_gt, '# pred2:', num_pred2)

    # deal with grey area
    sorted_pred = sorted(thresh_r_scores, key=lambda x:x[1], reverse=True)
    correct, num_pred = num_fixed_correct, num_fixed_pred
    precs, recalls = [], []
    for i, item in enumerate(sorted_pred):
        correct += item[0]
        num_pred += 1
        precs.append( correct / num_pred)  # Precision
        recalls.append( correct / num_gt)  # Recall

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]
    print('Best thresh', thresh, '\tbest F1', f1)

    for item in sorted_pred[:f1_pos]:
        # add to res
        tmp_dict = {
            'title': item[2],
            'h_idx': item[3],
            't_idx': item[4],
            'r': id2rel[item[5]],
        }
        res.append(tmp_dict)

    return res, thresh

def to_official(preds, features, sen_preds=[]):
    h_idx, t_idx, title = [], [], []

    if len(sen_preds) > 0:
        if len(sen_preds[0].shape) == 2:
            if_at = True
        elif len(sen_preds[0].shape) == 1:
            if_at = False

    for f in features:
        if 'original_hts' in f:
            hts = f['original_hts']
        else:
            hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

        if 'htbs' in f:
            htbs = f['htbs']
            h_idx += [ht[0][0] for ht in htbs]
            t_idx += [ht[0][1] for ht in htbs]
            title += [f["title"] for ht in htbs]

    res = []
    evi_by_title = {}
    num_pairs_with_evidence = 0
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        if len(sen_preds) > 0:
            if if_at:
                sen_pred = sen_preds[i] # sen_preds[i]: [num_sents, num_rels] or [num_sents]
                sen_pred = np.nonzero( np.sum(sen_pred[:,1:], axis=-1) )[0].tolist()
            else:
                sen_pred = np.nonzero(sen_preds[i])[0].tolist()

            if len(sen_pred) > 0:
                h,t,tit = h_idx[i], t_idx[i], title[i]
                if tit not in evi_by_title:
                    evi_by_title[tit] = {}
                evi_by_title[tit][(h,t)] = sen_pred
                num_pairs_with_evidence += 1

        for idx, p in enumerate(pred):
            if p != 0:
                tmp_dict = {
                    'title': title[i],
                    'h_idx': h_idx[i],
                    't_idx': t_idx[i],
                    'r': id2rel[p],
                }
                if len(sen_preds) > 0:
                    tmp_dict['evidence'] = sen_pred
                res.append(tmp_dict)

    if len(sen_preds) > 0:
        print('num of pairs with evidence:', num_pairs_with_evidence)

    if len(evi_by_title) > 0:
        return res, evi_by_title
    else:
        return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, tot_rel = -1, mode='dev'):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train_annotated.json"), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    if mode == 'dev':
        truth = json.load(open(os.path.join(path, "dev.json")))
    elif mode == 'train':
        truth = json.load(open(os.path.join(path, "train_annotated.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        # delete redundant items
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    if tot_rel > 0:
        tot_relations = tot_rel

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train
