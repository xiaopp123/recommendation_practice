#encoding=utf-8
from data_set import load_data
from data_set import FeatureDictionary, DataParser
from sklearn.model_selection import StratifiedKFold
import config
import tensorflow as tf
from DeepFM import DeepFM
from metrics import gini_norm
from time import time
import numpy as np
from sklearn.metrics import roc_auc_score


# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": 5,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}

def train_model(sess, model, Xi_train_, Xv_train_, y_train_, Xi_valid_=None, Xv_valid_=None, y_valid_=None):
    t1 = time()
    has_valid = Xv_valid_ is not None
    for epoch in range(dfm_params['epoch']):
        for i in range(0, len(Xi_train_), dfm_params['batch_size']):
            end = min(i + dfm_params['batch_size'], len(Xi_train_))
            Xi = Xi_train_[i: end]
            Xv = Xv_train_[i: end]
            y = y_train_[i: end]
            feed_dict = {
                model.feat_index: Xi,
                model.feat_value: Xv,
                model.label: np.expand_dims(y, axis=1),
                model.dropout_keep_fm: dfm_params['dropout_fm'],
                model.dropout_keep_deep: dfm_params['dropout_deep'],
                model.train_phase: True
            }
            loss, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)
        #验证
        if dfm_params['verbose'] > 0 and (epoch + 1) % dfm_params['verbose'] == 0:
            if has_valid:
                y_predict = test_model(sess, model, Xi_valid_, Xv_valid_, y_valid_)
                valid_result = roc_auc_score(y_valid_, y_predict)
                print("[%d] valid-result=%.4f [%.1f s]" % (epoch + 1, valid_result, time() - t1))

def test_model(sess, model, Xi_test, Xv_test, y_test):
    y_pred = None
    for i in range(0, len(Xi_test), dfm_params['batch_size']):
        end = min(i + dfm_params['batch_size'], len(Xi_test))
        Xi = Xi_test[i: end]
        Xv = Xv_test[i: end]
        y = y_test[i: end]
        feed_dict = {
             model.feat_index: Xi,
             model.feat_value: Xv,
             model.label: np.expand_dims(y, axis=1),
             model.dropout_keep_fm: [1.0] * len(dfm_params['dropout_fm']),
             model.dropout_keep_deep: [1.0] * len(dfm_params['dropout_deep']),
             model.train_phase: False
        }
        logits = sess.run([model.out], feed_dict=feed_dict)
        #最后一个batch大小可能与之前的不一样
        if i == 0:
            y_pred = np.reshape(logits[0], [-1, 1])
        else:
            y_pred = np.concatenate((y_pred, np.reshape(logits[0], [-1, 1])))
    return y_pred
        

if __name__ == '__main__':
    train_df, test_df, x_train, y_train, x_test, ids_test, cat_features_indices = load_data()
    print(train_df)
    # k折交叉验证
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,\
                             random_state=config.RANDOM_SEED).split(x_train, y_train))

    fd = FeatureDictionary(dfTrain=train_df, dfTest=test_df,\
                           numeric_cols=config.NUMERIC_COLS,\
                           ignore_cols=config.IGNORE_COLS)

    #
    data_parser = DataParser(feat_dict=fd)
    #Xi_train训练数据中特征编号,Xv_train是Xi_train对应的特征编号下的特征值，连续型特征则是该特征值
    Xi_train, Xv_train, y_train = data_parser.parse(df=train_df, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=test_df)

    _get = lambda x, l: [x[i] for i in l]

    dfm_params['feature_size'] = fd.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])
    dfm = DeepFM(**dfm_params)

    y_test_meta = np.zeros((x_test.shape[0], 1), dtype=float)
    with tf.Session() as sess:
        summ_writer = tf.summary.FileWriter('summaries', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i, (train_idx, valid_idx) in enumerate(folds):
            print('%d fold' % i)
            #训练数据
            Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
            #验证数据
            Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
            #训练模型并进行验证
            train_model(sess, dfm, Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
            #测试
            y_test = test_model(sess, dfm, Xi_test, Xv_test, [1] * len(Xi_test))
            #K折交叉的最好结果是K次结果的平均
            y_test_meta += y_test

        y_test_meta /= float(len(folds))
