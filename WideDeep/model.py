#encoding=utf-8
import tensorflow as tf

def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
    wide_columns, deep_columns = model_column_fn()
    hidden_units = [100, 75, 50, 25]
    #why CPU is faster than GPU 
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0},
                                      inter_op_parallelism_threads=inter_op,
                                      intra_op_parallelism_threads=intra_op))
    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)

def export_model(model, model_type, export_dir, model_column_fn):
    """Export to SavedModel format.
    
    Args:
      model: Estimator object
      model_type: string indicating model type. "wide", "deep" or "wide_deep"
      export_dir: directory to export the model.
      model_column_fn: Function to generate model feature columns.
    """
    wide_columns, deep_columns = model_column_fn()
    if model_type == 'wide':
        columns = wide_columns
    elif model_type == 'deep':
        columns = deep_columns
    else:
        columns = wide_columns + deep_columns

    feature_spec = tf.feature_column.make_parse_example_spec(columns)

    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))

    model.export_savedmodel(export_dir, example_input_fn,
                            strip_default_attrs=True)
