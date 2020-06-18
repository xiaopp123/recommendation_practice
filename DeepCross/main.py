#encoding=utf-8
import argparse
import shutil
import sys

import tensorflow as tf

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', type=str, default='./',
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default='deep_cross',
    help="Valid model types: {'wide', 'deep', 'wide_deep', 'deep_cross'}.")
parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=64, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default='../MLR/data/adult.data.txt',
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default='../MLR/data/adult.test.txt',
    help='Path to the test data.')
parser.add_argument(
    '--num_cross_layers', type=int, default=4,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--hidden_units', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

FLAGS, unparsed = parser.parse_known_args()

#指定列的类型
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

def build_model_columns():
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    education = tf.feature_column.categorical_column_with_vocabulary_list(
            'education', [
                'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
            'marital_status', [
                'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
            'relationship', [
                'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
            'workclass', [
                'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
                'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
            'occupation', hash_bucket_size=1000)
    columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(occupation)]
    return columns

def cross_layer2(x0, x, name):
    with tf.variable_scope(name):
        input_dim = x0.get_shape().as_list()[1]
        w = tf.get_variable('weight', [input_dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('bias', [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        xw = tf.matmul(x, w) #[batch, 1]
        x0xw = tf.multiply(x0, xw) #[batch, input_dim]
        return x0xw + b + x #[batch, input_dim]

def build_cross_layers(x0, params):
    num_layers = FLAGS.num_cross_layers
    x = x0
    for i in range(num_layers):
        x = cross_layer2(x0, x, 'cross_{}'.format(i))
    return x

def build_deep_layers(x0, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.
    #batch normalize
    net = x0
    #使用了两个隐层
    hidden_units = [1024, 1024] 
    #to do
    for units in hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    return net

def build_model(features, labels, mode, params):
    columns = build_model_columns()
    #对特征编码, 这里没有对one-hot向量使用embedding
    input_layer = tf.feature_column.input_layer(features = features, feature_columns = columns)
    
    #deep和cross参考 https://blog.csdn.net/Dby_freedom/article/details/86502623
    #cross层
    last_cross_layer = build_cross_layers(input_layer, params)
    
    #deep层
    last_deep_layer =  build_deep_layers(input_layer, params)

    #cross和deep的输出进行concate
    last_layer = tf.concat([last_cross_layer, last_deep_layer], 1)
    
    '''
    my_head = tf.estimator.BinaryClassHead(thresholds=0.5)

    logits = tf.layers.dense(last_layer, units=my_head.logits_dimension, activation=None, use_bias=True)
    
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    '''

    logits = tf.layers.dense(last_layer, units=1, activation=None, use_bias=True)
    out_probs = tf.nn.sigmoid(logits)
    predictions = tf.cast(out_probs > 0.5, tf.float32, name='predict_labels')
    
    labels = tf.cast(labels, tf.float32, name='true_labels')
    
    #定义损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    accuracy = tf.metrics.accuracy(labels, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
    train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        print('ERROR')

def build_estimator(model_dir, model_type):
    """
    使用tensorflow高阶API,estimator
    """
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'CPU': 6}))

    if model_type == 'deep_cross':
        return tf.estimator.Estimator(model_fn = build_model, config=run_config)
    else:
        print ('error')
     

def input_fn(data_file=FLAGS.train_data, num_epochs=FLAGS.epochs_per_eval, shuffle=True, batch_size=FLAGS.batch_size):
    def process_list_column(list_column):
        sparse_strings = tf.string_split(list_column, delimiter="|")
        return sparse_strings.values

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        features['workclass'] = process_list_column([features['workclass']])
        labels = tf.equal(features.pop('income_bracket'), ' >50K')
        labels = tf.reshape(labels, [-1])
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=15000)
    #num_parallel_calls并行计算
    dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    #element = iterator.get_next()
    '''
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(element))
    '''
    return features, labels


if __name__ == '__main__':
    #input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size)
    
    #构建模型
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        #lambda的意义
        model.train(input_fn=lambda:
                input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))
        results = model.evaluate(input_fn=lambda:
                input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size))
        # 显示evaluation中的衡量指标
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
