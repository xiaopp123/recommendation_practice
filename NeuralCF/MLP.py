#encoding=utf-8

import tensorflow as tf
import argparse
from time import time
from Dataset import Dataset
from MLP_Model import MLP_Model
import numpy as np
import os
from evaluate import evaluate_model

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='neural_collaborative_filtering/Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    top_k = 10
    evaluation_threads = 1

    #loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train_data, test_ratings, test_negatives = dataset.train_matrix, dataset.test_ratings, dataset.test_negatives
    num_users, num_items = train_data.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train_data.nnz, len(test_ratings)))

    #build MLP model
    mlp_model = MLP_Model(num_users, num_items, learning_rate, learner, layers)
    mlp_model.build_graph()

    #获取每个训练测试实例
    user_input, item_input, labels = get_train_instances(train_data, num_negatives)

    if not os.path.exists('summaries'):
        os.mkdir('summaries')

    #记录在测试集上最好的命中率(best_hr)
    best_hr, best_ndcg, best_iter = None, None, None

    with tf.Session() as sess:
        summ_writer = tf.summary.FileWriter('summaries', sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            t1 = time()
            epoch_loss = 0.0
            for i in range(0, len(user_input), batch_size):
                end = min(i + batch_size, len(user_input))
                feed_dict = {
                    mlp_model.user_inputs: user_input[i: end],
                    mlp_model.item_inputs: item_input[i: end],
                    mlp_model.y: labels[i: end]
                }
                train_op, logits, loss = sess.run([mlp_model.train_op, mlp_model.logits, mlp_model.loss], feed_dict=feed_dict)
                epoch_loss += loss
            t2 = time()
            #验证
            if (epoch + 1) % verbose == 0:
                hits, ndcgs = evaluate_model(mlp_model, sess, test_ratings, test_negatives, top_k, evaluation_threads)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                    % (epoch,  t2 - t1, hr, ndcg, epoch_loss, time() - t2))

                if best_hr is None or hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        print("save model")
                #save summary 
                feed_dict = {
                    mlp_model.tf_loss_ph: epoch_loss,
                    mlp_model.tf_hr_ph: hr,
                    mlp_model.tf_ndcg_ph: ndcg
                }
                summ = sess.run(mlp_model.performance_summaries, feed_dict=feed_dict)
                summ_writer.add_summary(summ, epoch)

        print("End.Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
