#encoding=utf-8
import argparse
import tensorflow as tf
from Dataset import Dataset
from time import time
from Neural_Model import Neural_Model
import numpy as np
import os
from evaluate import evaluate_model

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='neural_collaborative_filtering/Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
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
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    top_k = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF arguments: %s " %(args))

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train_data, test_ratings, test_negatives = dataset.train_matrix, dataset.test_ratings, dataset.test_negatives
    num_users, num_items = train_data.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train_data.nnz, len(test_ratings)))

    neural_model = Neural_Model(num_users, num_items, mf_dim, layers, learning_rate, learner)
    neural_model.build_graph()

    #获取每个训练测试实例
    user_input, item_input, labels = get_train_instances(train_data, num_negatives)

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries', 'NeuralMF')):
        os.mkdir(os.path.join('summaries', 'NeuralMF'))

    #记录在测试集上最好的命中率(best_hr)
    best_hr, best_ndcg, best_iter = None, None, None

    with tf.Session() as sess:
        summ_writer = tf.summary.FileWriter('summaries/NeuralMF', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            t1 = time()
            epoch_loss = 0.0
            for i in range(0, len(user_input), batch_size):
                end = min(i + batch_size, len(user_input))
                feed_dict = {
                    neural_model.user_inputs: user_input[i: end],
                    neural_model.item_inputs: item_input[i: end],
                    neural_model.y: labels[i: end]
                }
                train_op, logits, loss = sess.run([neural_model.train_op, neural_model.logits, neural_model.loss], feed_dict=feed_dict)
                epoch_loss += loss
            t2 = time()
            #验证
            if (epoch + 1) % verbose == 0:
                hits, ndcgs = evaluate_model(neural_model, sess, test_ratings, test_negatives, top_k, evaluation_threads)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                    % (epoch,  t2 - t1, hr, ndcg, epoch_loss, time() - t2))

                if best_hr is None or hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        print("save model")
                #save summary 
                feed_dict = {
                    neural_model.tf_loss_ph: epoch_loss,
                    neural_model.tf_hr_ph: hr,
                    neural_model.tf_ndcg_ph: ndcg
                }
                summ = sess.run(neural_model.performance_summaries, feed_dict=feed_dict)
                summ_writer.add_summary(summ, epoch)

        print("End.Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
