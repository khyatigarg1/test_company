import json
import string
import os
from datetime import datetime
from glob import glob

# note: tfio must be imported before fsspec see: https://github.com/tensorflow/io/issues/684
# import tensorflow_io as tfio
import fsspec
import pandas
import nmslib
import numpy
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Lambda, Conv1D, GlobalMaxPool1D, concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
import toml
import flatten_dict
from tensorboard.plugins.hparams import api as hp
from sklearn.metrics.pairwise import paired_cosine_distances
from model.evaluate_emb import evaluate


import posixpath
import deepmerge
from copy import deepcopy



def train_task(task_definition):
    task_definition = AttrDict(task_definition)

    dataset_path = task_definition['input']['filter_dataset']['final_path']
    compres_db = os.path.join(task_definition['input']['compres_data']['final_path'], 'data.json')
    train_output_path = task_definition['output']['train']['final_path']
    model_output_path = task_definition['output']['model']['final_path']


    model = CompresModel(task_definition.MODEL)
    model = train(model, dataset_path, train_output_path, task_definition.TRAIN, 'parquet')
    model.save(model_output_path)

    accuracy = evaluate(train_output_path, model, compres_db)
    if accuracy < task_definition.TRAIN.MIN_ACCURACY:
        raise Exception(f'Model accuracy {accuracy} below threshold')


# MODEL DEFINITION
class CompresModel:
    def __init__(self, model_config):
        self._model_config = model_config
        self._model = get_model(self._model_config)

    @staticmethod
    def load(model_dir, model_config):
        # model_config = AttrDict(toml.load(open(os.path.join(model_dir, 'model_config.toml'))))
        model = CompresModel(model_config)
        model._model.load_weights(os.path.join(model_dir, 'model.keras'))
        return model

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        toml.dump(self._model_config, open(os.path.join(model_dir,'model_config.toml'), 'w'))
        self._model.save_weights(os.path.join(model_dir, 'model.keras'))

    # @Timer('CompresModel.predict')
    def predict(self, inputs):
        return self._model.predict(inputs)

    def fit(self, inputs):
        pass

    def transform(self, inputs):
        return self.predict(inputs)

# def get_char_map(is_case_sensitive):
#     """ Build a tensorflow character lookup layer to map from character to numeric id"""
#     chars = string.punctuation + string.digits + string.whitespace + string.ascii_lowercase
#     if is_case_sensitive:
#         chars += string.ascii_uppercase
#     return tf.lookup.StaticHashTable(
#         tf.lookup.KeyValueTensorInitializer(list(chars), list(range(1, len(chars) + 1)), 'string', 'int32'), 0)


def string_to_vec(text_tensor, is_case_sensitive, min_ngram, max_ngram, num_buckets):
    """ map the input string to tensor of character ids"""
    if not is_case_sensitive:
        text_tensor = tf.strings.lower(text_tensor)
    char_tensor = tf.strings.unicode_split(text_tensor, 'UTF-8')
    ngrams_tensor = tf.strings.ngrams(char_tensor, ngram_width=list(range(min_ngram, max_ngram+1)), separator='', preserve_short_sequences=True)
    token_ids_tensor = tf.strings.to_hash_bucket_strong(ngrams_tensor, num_buckets=num_buckets, key=[42, 72])
    dense_tensor = token_ids_tensor.to_tensor()
    return dense_tensor


@tf.function
def loss_function(y_true, y_pred):
    """
    The loss function for training
    """
    success_rate, same_class, different_class = metric_function(y_pred)
    return different_class - same_class


@tf.function
def same_class(y_true, y_pred):
    """ metric function - the average similarity of aliases for the same company"""
    success_rate, same_class, different_class = metric_function(y_pred)
    return same_class


@tf.function
def different_class(y_true, y_pred):
    """ metric function - the average similarity of aliases for different companies"""
    success_rate, same_class, different_class = metric_function(y_pred)
    return different_class


@tf.function
def success_rate(y_true, y_pred):
    """ metric function - the success rate of matching aliases within one batch """

    success_rate, same_class, different_class = metric_function(y_pred)
    return success_rate

@tf.function
def metric_function(y_pred):
    """
    the function that generates the data for the loss and metric functions
    """
    n = tf.keras.backend.shape(y_pred)[0]//2
    # print("I'm here!!!")
    # print(y_pred)
    y_pred = tf.math.l2_normalize(y_pred, 1)
    # print(y_pred)

    a = y_pred[0::2]
    b = y_pred[1::2]

    all_pairs_cosine_similarity = tf.matmul(a, b, transpose_b=True)
    # print(all_pairs_cosine_similarity)

    success_rate = tf.reduce_mean(
        tf.cast(tf.range(n) == tf.argmax(all_pairs_cosine_similarity, output_type='int32'), 'float32'))


    same_class_mask = tf.eye(n)

    # TODO consider replacing mean with min\max to compute loss based on worse result instead of average result
    # same_class = tensorflow.reduce_min(tensorflow.boolean_mask(all_pairs_cosine_similarity, same_class_mask))
    # different_class = tensorflow.reduce_max(tensorflow.boolean_mask(all_pairs_cosine_similarity, 1-same_class_mask))
    same_class = tf.reduce_mean (tf.boolean_mask(all_pairs_cosine_similarity, same_class_mask))
    different_class = tf.reduce_mean(tf.boolean_mask(all_pairs_cosine_similarity, 1-same_class_mask))
    return success_rate, same_class, different_class


def get_embedding_layer(model_config):
    """ build a character embedding layer"""
    return Embedding(input_dim=model_config['CHAR_EMBEDDING_INPUT_DIM'],
                     output_dim=model_config['CHAR_EMBEDDING_OUTPUT_DIM'])


def bag_of_words_submodel(model_config):
    return GlobalAveragePooling1D()


def lstm_submodel(model_config):
    """ build a sequence embedding submodel based on LSTM"""
    return Bidirectional(LSTM(model_config['SEQUENCE_ENCODER_DIM'], activation='relu'))
    # for test purposes - check also sigmoid
    # return Bidirectional(LSTM(model_config['SEQUENCE_ENCODER_DIM'], activation='sigmoid'))


# def cnn_submodel(model_config):
#     """ build a sequence embedding submodel based on CNN"""
#     embedding = get_embedding_layer(model_config)(input_tensor)
#     l1 = concatenate([Conv1D(500, kernel_size, activation='relu', padding='same')(embedding) for kernel_size in range(1,10)])
#     max_pool = GlobalMaxPool1D()(l1)
#     return max_pool
#
#
# def transformer_submodel(model_config):
#     """ build a sequence embedding submodel based on a transformer"""
#
#     from official.transformer.v2.transformer import EncoderStack, Transformer
#     from official.transformer.model import model_utils
#
#     transfomer_params = {
#         'num_hidden_layers': 3,
#         'hidden_size': model_config.CHAR_EMBEDDING_OUTPUT_DIM,
#         'num_heads': 10,
#         'attention_dropout': 0.3,
#         'filter_size': 300,
#         'relu_dropout': 0.3,
#         'layer_postprocess_dropout': 0.3,
#         'vocab_size': model_config.CHAR_EMBEDDING_INPUT_DIM,
#         'dtype': 'float32',
#     }
#     training = tf.keras.backend.learning_phase()
#     attention_bias = model_utils.get_padding_bias(input_tensor)
#     transfomer = Transformer(transfomer_params)
#     encoded = transfomer.encode(input_tensor, attention_bias, training)
#     flattened = Flatten()(encoded)
#     return flattened


SEQUENCE_LAYERS = {
    # 'cnn': cnn_submodel,
    'lstm': lstm_submodel
    # 'transformer': transformer_submodel,
    # 'bow': bag_of_words_submodel
}


def get_model(model_config):
    """ build a model object"""
    input_layer = Input(shape=tuple(), ragged=False, dtype='string')

    feature_layer = Lambda(string_to_vec,
                           arguments={'is_case_sensitive': model_config['IS_CASE_SENSITIVE'],
                                      'min_ngram': model_config['NGRAM_MIN'],
                                      'max_ngram': model_config['NGRAM_MAX'],
                                      'num_buckets': model_config['CHAR_EMBEDDING_INPUT_DIM']
                                      })(input_layer)
    embedding = get_embedding_layer(model_config)(feature_layer)

    sequence_layer = SEQUENCE_LAYERS[model_config['SEQUENCE_ENCODER']](model_config)(embedding)
    if model_config['USE_DENSE']:
        output_layer = Dense(model_config['DENSE_DIM'], activation='relu')(sequence_layer)
    else:
        output_layer = sequence_layer

    # output_layer = Lambda(tf.math.l2_normalize, arguments={'axis': 1})(output_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer=Adam(), loss=loss_function, metrics=[success_rate, same_class, different_class])
    # model.compile(optimizer=Adam(), loss=loss_function, metrics=['accuracy'])
    return model

# TRAIN


def read_dataset(path, format, batch_size):

    if format == 'csv':
        return tf.data.experimental.make_csv_dataset(path, batch_size=batch_size)
    if format == 'parquet':
        df = pandas.read_parquet(path)
        print(df.to_dict('list'))
        # For debugging (none values)
        # df_dict={}
        # df_dict['a'] = df.to_dict('list')['a']=df.to_dict('list')['a'][925:950]
        # df_dict['b'] = df.to_dict('list')['b'] = df.to_dict('list')['b'][925:950]
        # ds = tf.data.Dataset.from_tensor_slices(df_dict)
        ds = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))
        return ds.batch(batch_size, drop_remainder=True)
    if format == 'tfio_parquet':
        import tensorflow_io as tfio
        files = [f's3://{f}' for f in fsspec.filesystem('s3').glob(path)] if path.startswith('s3:') else glob(path)
        data_sets = [tfio.IODataset.from_parquet(f) for f in files]
        # workaround for - https://github.com/tensorflow/io/issues/677
        data_sets = [ds.map(lambda x,y: {'a': x, 'b': y}) for ds in data_sets]
        merged_data_set = tf.data.Dataset.from_tensor_slices(data_sets).interleave(lambda x:x)
        return merged_data_set.batch(batch_size, drop_remainder=True)
    raise Exception(f'unsuported format {format}')


def interleave_dataset(dataset, batch_size):
    """ interleave a dataset to make it suitable as input to the model
    a1,b1
    a2,b2
    ...
    to:
    a1
    b1
    a2
    b2

    """
    # batch_size = dataset._flat_shapes[0].as_list()[0]
    # return dataset.map(lambda batch: tf.dynamic_stitch([range(0,2*batch_size,2),range(1,2*batch_size,2)],list(batch.values())))
    return dataset.map(lambda batch: tf.dynamic_stitch([list(range(0,2*batch_size,2)),list(range(1,2*batch_size,2))],list(batch.values())))


def train(model, dataset_file, output_dir, train_config, file_format):
    """ train the model"""
    # if tf.__version__ >= '2.1.0':
    #     tf.debugging.enable_check_numerics()

    os.makedirs(output_dir, exist_ok=True)
    dataset = read_dataset(dataset_file, file_format, train_config['BATCH_SIZE'])
    dataset = dataset.shuffle(1000)
    dataset = interleave_dataset(dataset, train_config['BATCH_SIZE'])
    # add dummy label column
    dataset = dataset.map(lambda x: (x, tf.zeros((1, 1))))

    # for x, y in dataset:
    #     print(x, '--------', y)

    test_dataset = dataset.take(train_config['TEST_SET_SIZE']//train_config['BATCH_SIZE'])
    train_dataset = dataset.skip(train_config['TEST_SET_SIZE']//train_config['BATCH_SIZE']).repeat()

    history = model._model.fit(train_dataset,validation_data=test_dataset, steps_per_epoch=train_config['STEPS_PER_EPOCH'], epochs=train_config['EPOCHS'], verbose=2, callbacks=[
        TerminateOnNaN(),
        ModelCheckpoint(filepath=os.path.join(output_dir, 'CP_{epoch:02d}.keras'), save_weights_only=True),
        TensorBoard(log_dir=output_dir,
                    histogram_freq=1,
                    write_graph=False,
                    embeddings_freq=1,
                    ),
        hp.KerasCallback(output_dir, flatten_dict.flatten({'train': train_config, 'model': model._model_config}, 'path'))
    ])
    if any(numpy.isnan(loss) for loss in history.history['loss']):
        raise Exception('failed traininig - NaN loss')
    return model










task_definition={}
task_definition['entry_point'] = "seo.compres_model.model:train_task"
task_definition['local_cache']='/tmp'

task_definition['defaults']={}
task_definition['defaults']['flow_name'] = "compres_model"
task_definition['defaults']['task_name'] = "train"
task_definition['defaults']['tag'] = "v1"
task_definition['defaults']['root'] = "s3://S3-BUCKET-NAME/taskwrapper"

task_definition['input']={}
task_definition['input']['filter_dataset']={}
task_definition['input']['filter_dataset']['sync']=True
task_definition['input']['filter_dataset']['flow_id']='*'
task_definition['input']['filter_dataset']['task_name']='filter'

task_definition['input']['compres_data']={}
task_definition['input']['compres_data']['sync']=True
task_definition['input']['compres_data']['resolved_path']='s3://dev-nosensitive-BUCKET-MYTEAM/train_task/compres/compres_train_input/compres/'

task_definition['output']={}
task_definition['output']['model']={}
task_definition['output']['model']['sync']=True

task_definition['output']['train']={}
task_definition['output']['train']['sync']=True
task_definition['output']['train']['sync_failed']=True

task_definition['MODEL']={}
task_definition['MODEL']['MAX_STRING_LEN'] = 100
task_definition['MODEL']['CHAR_EMBEDDING_INPUT_DIM'] = 400 #499
task_definition['MODEL']['CHAR_EMBEDDING_OUTPUT_DIM'] = 400
task_definition['MODEL']['SEQUENCE_ENCODER_DIM'] = 400
task_definition['MODEL']['DENSE_DIM'] = 500
task_definition['MODEL']['NGRAM_MAX'] = 3
task_definition['MODEL']['NGRAM_MIN'] = 1
# task_definition['MODEL']['DROPOUT'] = 0.25

# lstm, cnn, transfomer, bow
task_definition['MODEL']['SEQUENCE_ENCODER'] = 'lstm'
task_definition['MODEL']['USE_DENSE'] = False
task_definition['MODEL']['IS_CASE_SENSITIVE'] = False

task_definition['TRAIN']={}
task_definition['TRAIN']['BATCH_SIZE'] = 10 #20
task_definition['TRAIN']['TEST_SET_SIZE'] = 1650 #1000 #100 #1000
task_definition['TRAIN']['EPOCHS'] = 2 #10 #4 #20 #4 #40
task_definition['TRAIN']['STEPS_PER_EPOCH'] = 10 #50 #200 #50 #500
task_definition['TRAIN']['MIN_ACCURACY'] = 0.7

task_definition['input']['filter_dataset']['final_path'] = '/Users/ranziv/Downloads/cem-output/datasetFiltering_output-no-fuzzy/filtered--0.csv'

task_definition['input']['compres_data']['final_path'] = '/Users/ranziv/Downloads/'
train_output_path = task_definition['output']['train']['final_path'] = '/Users/ranziv/Downloads/cem-output/train_output/'
model_output_path = task_definition['output']['model']['final_path'] = '/Users/ranziv/Downloads/cem-output/model_output/'

dataset_path = task_definition['input']['filter_dataset']['final_path']
compres_db = os.path.join(task_definition['input']['compres_data']['final_path'], 'data-f1k.json')
train_output_path = task_definition['output']['train']['final_path']
model_output_path = task_definition['output']['model']['final_path']


model = CompresModel(task_definition['MODEL'])
# model = train(model, dataset_path, train_output_path, task_definition['TRAIN'], 'parquet')

# to train:
# ---------
model = train(model, dataset_path, train_output_path, task_definition['TRAIN'], 'csv')
# model.save(model_output_path)

# to test:
# --------
model.load(model_output_path, task_definition['MODEL'])
stats_dict={}

# accuracy, stats_dict = evaluate(train_output_path, model, compres_db)
# if accuracy < task_definition['TRAIN']['MIN_ACCURACY']:
#     raise Exception(f'Model accuracy {accuracy} below threshold')

# samples_path = '/Users/ranziv/Downloads/cem-input/data-f1k-samples/'
# amount_of_samples = 100
samples_path = '/Users/ranziv/Downloads/cem-input/data-f1k-iterations/'
amount_of_samples = 10


comp2vec_accuracy = {1:[], 2:[], 3:[]}
fuzzy_accuracy = {1:[], 2:[], 3:[]}
for i in range(1, amount_of_samples+1):
    print('---- iteration # ' + str(i) + ' is executed ----')
    # samples_path = '/Users/ranziv/Downloads/cem-input/data-f1k-samples/'
    sample_fname = samples_path + 'data-f1k-sample-' + str(i) + '.json'
    accuracy, stats_dict = evaluate(train_output_path, model, sample_fname, stats_dict, i)
    comp2vec_accuracy[1].append(stats_dict[i]['Accuracy Match @ 1'])
    comp2vec_accuracy[2].append(stats_dict[i]['Accuracy Match @ 2'])
    comp2vec_accuracy[3].append(stats_dict[i]['Accuracy Match @ 3'])
    fuzzy_accuracy[1].append(stats_dict[i]['Accuracy Fuzzy Distance Match @ 1'])
    fuzzy_accuracy[2].append(stats_dict[i]['Accuracy Fuzzy Distance Match @ 2'])
    fuzzy_accuracy[3].append(stats_dict[i]['Accuracy Fuzzy Distance Match @ 3'])


print(accuracy)
print(stats_dict)

print(comp2vec_accuracy)
print(fuzzy_accuracy)
