#encoding=utf-8
import argparse
import shutil
import sys
import tensorflow as tf
from model import build_estimator, export_model
from data_utils import build_model_columns as model_column_fn
from data_utils import input_fn

#tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', type=str, default='./census_model',
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep', 'deep_cross'}.")
parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default='./datas/adult.data.clean',
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default='./datas/adult.test.clean',
    help='Path to the test data.')
parser.add_argument(
    '--num_cross_layers', type=int, default=4,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--hidden_units', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--inter_op_parallelism_threads', type=int, default=0,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--intra_op_parallelism_threads', type=int, default=0,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--export_dir', type=str, default='./exported_model',
    help='Path to the test data.')

FLAGS, unparsed = parser.parse_known_args()

def train_input_fn():
    return input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size)

def eval_input_fn():
    return input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size)
   

def main():
    model = build_estimator(
        model_dir=FLAGS.model_dir, model_type=FLAGS.model_type,
        model_column_fn=model_column_fn,
        inter_op=FLAGS.inter_op_parallelism_threads,
        intra_op=FLAGS.intra_op_parallelism_threads
    )
    
    cnt = 0
    for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=train_input_fn)
        results = model.evaluate(input_fn=eval_input_fn)
        print(results)
        '''
        if FLAGS.early_stop and FLAGS.stop_threshold > results['accuracy']
            cnt += 1
        else:
            cnt = 0
        if cnt > 5:
            break
        '''

    if FLAGS.export_dir is not None:
        export_model(model, FLAGS.model_type, FLAGS.export_dir, model_column_fn)

if __name__ == '__main__':
    main()
