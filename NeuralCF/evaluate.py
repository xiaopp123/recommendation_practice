#encoding=utf-8
import numpy as np
import math
import heapq

def evaluate_model(model, sess, test_rating, test_negativates, k, num_thread):
    hits, ndcgs = [], []
    for idx in range(len(test_rating)):
        hr, ndcg = eval_one_rating(model, sess, test_rating[idx], test_negativates[idx], k)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs

def eval_one_rating(model, sess, rating, negativates, k):
    u = rating[0]
    gitem = rating[1]
    items = negativates
    items.append(gitem)
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    labels = np.full(len(items), 0, dtype='int32')
    feed_dict = {
        model.user_inputs: users,
        model.item_inputs: items,
        model.y: labels
    }
    predictions = sess.run([model.logits], feed_dict=feed_dict)[0]
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    #necessory?
    #items.pop()
    #evaluate topk
    ranklist = heapq.nlargest(k, map_item_score.items(), key=lambda x: x[1])

    hr = get_hit_ratio(ranklist, gitem)
    ndcg = get_ndcg(ranklist, gitem)

    return hr, ndcg

def get_hit_ratio(ranklist, gitem):
    for item in ranklist:
        if item[0] == gitem:
            return 1
    return 0

def get_ndcg(ranklist, gitem):
    for i, item in enumerate(ranklist):
        if item[0] == gitem:
            return math.log(2) / math.log(i + 2)
    return 0
