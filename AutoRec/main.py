#encoding=utf-8
import tensorflow as tf
import time
import argparse
from data_process import read_rating
from model import AutoRec
import numpy as np

current_time = time.time()

parser = argparse.ArgumentParser(description='I-AutoRec')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=100)

parser.add_argument('--optimizer_method', choices=['Adam','RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()

file_name = '../ml-100k/u.data'
num_user = 943
num_item = 1682
num_total_ratings = 100000
train_ratio = 0.8


train_rating, train_mask_rating, test_rating, test_mask_rating, user_train_set, item_train_set, user_test_set, item_test_set =\
    read_rating(file_name, num_user, num_item, num_total_ratings, train_ratio)

with tf.Session() as sess:
    auto_rec = AutoRec(num_item, num_user, args) 
    auto_rec.build_graph()
    #注意初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #对训练数据shuffle
    random_perm_doc_idx = np.random.permutation(np.arange(num_user))
    train_loss = []
    test_loss = []
    start_time = time.time()

    for e in range(args.train_epoch):
        epoch_loss = 0
        for i in range(0, num_user, args.batch_size):
            end = min(i + args.batch_size, num_user)
            batch_set_idx = random_perm_doc_idx[i: end]
            feed_dict = {
                auto_rec.input_rating: train_rating[batch_set_idx, :],
                auto_rec.input_mask_rating: train_mask_rating[batch_set_idx, :]
            }
            _, cost = sess.run([auto_rec.optimizer, auto_rec.cost], feed_dict=feed_dict)
            epoch_loss += cost
        train_loss.append(epoch_loss)
        #验证集
        if (e + 1 ) % 50 == 0:
             #验证的时候不是使用batch，而是全部数据
             feed_dict = {
                 auto_rec.input_rating: test_rating,
                 auto_rec.input_mask_rating: test_mask_rating
             }
             cost, decoder = sess.run([auto_rec.cost, auto_rec.decoder], feed_dict=feed_dict)
             #rating的取值范围是[1,5]
             estimated_rating = decoder.clip(min=1, max=5)
             #仅预测测试集中未出现的user及item
             #从测试集中去掉在训练集中出现的user
             unseen_user_test_list = list(user_test_set - user_train_set)
             #从测试集中去掉在训练集中出现的item
             unseen_item_test_list = list(item_test_set - item_train_set)

             for user in unseen_user_test_list:
                 for item in unseen_item_test_list:
                     if self.test_mask_R[user, item] == 1: # exist in test set
                         estimated_rating[user, item] = 3 #统一设为3

             #求测试数据中的均方误差
             pre_numerator = tf.multiply(estimated_rating - test_rating, test_mask_rating).eval()
             numerator = np.sum(np.square(pre_numerator))
             denominator = len(test_rating)
             RMSE = np.sqrt(numerator / float(denominator))

             test_loss.append(RMSE)
             print ("Testing //", "Epoch %d //" % (e), " Total cost = {:.2f}".format(cost), " RMSE = {:.5f}".format(RMSE),
                    "Elapsed time : %d sec" % (time.time() - start_time))
             print ("=" * 100)
