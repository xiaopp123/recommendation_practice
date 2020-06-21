#encoding=utf-8

import tensorflow as tf
import argparse
from time import time
from Dataset import Dataset
from GMF_Model import GMF_Model
import numpy as np
import os


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='neural_collaborative_filtering/Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
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
    #input_perm = np.random.permutation(len(user_input))
    #users, items, labels_random = [], [], []
    #for i in input_perm:
    #    users.append(user_input[i])
    #    items.append(item_input[i])
    #    labels_random.append(labels[i])

    #return users, items, labels_random
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    #loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train_data, test_ratings, test_negatives = dataset.train_matrix, dataset.test_ratings, dataset.test_negatives
    num_users, num_items = train_data.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train_data.nnz, len(test_ratings)))

    gmf_model = GMF_Model(num_users, num_items, num_factors, regs, learning_rate, learner) 
    gmf_model.build_graph()

    user_input, item_input, labels = get_train_instances(train_data, num_negatives)

    if not os.path.exists('summaries'):
        os.mkdir('summaries')

    with tf.Session() as sess:
        summ_writer = tf.summary.FileWriter('summaries', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(user_input), batch_size):
                end = min(i + batch_size, len(user_input))
                feed_dict = {
                    gmf_model.user_inputs: user_input[i: end],
                    gmf_model.item_inputs: item_input[i: end],
                    gmf_model.y: labels[i: end]
                }
                #print(user_input[i:end], item_input[i: end], labels[i: end])
                train_op, logits, loss = sess.run([gmf_model.train_op, gmf_model.logits, gmf_model.loss], feed_dict=feed_dict)
                #print(train_op)
                epoch_loss += loss
            print(epoch_loss)
