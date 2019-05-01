"""
Deprecated: Prototyping and testing of a feed forward network in scikit learn. In the end, tensorflow was used.
"""
import os
import os.path
import pickle
import sys
from functools import partial
from multiprocessing import Manager, Pool
import joblib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

# get_ipython().run_line_magic('matplotlib', 'inline')

try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold as KFold

from thesis.util import save_pickle as save, load_pickle as load


def fully_connected(x, target_size, activation_function=None, normed=False, bias_init=0.0):
    fan_in = int(x.shape[-1])

    if activation_function == tf.nn.relu:
        var_init = tf.random_normal_initializer(stddev=2/fan_in)
    else:
        var_init = tf.random_normal_initializer(stddev=fan_in**(-1/2))
    weights = tf.get_variable("weights", [x.shape[1], target_size], tf.float32, var_init)

    var_init = tf.constant_initializer(bias_init)
    biases = tf.get_variable("biases", [target_size], tf.float32, var_init)

    activation = tf.matmul(x, weights) + biases

    return activation_function(activation) if callable(activation_function) else activation

def make_dataset(X_data, y_data, n_splits):    
    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data, y_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train.todense(), X_test.todense(), y_train, y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64, tf.float64, tf.float64, tf.float64))


frame = load("data/out/data")
cols_to_drop = []
for c in frame.columns:
    d = frame[c]
    if (d == -1).sum() > 0.5*frame.shape[0]:
        cols_to_drop.append(c)
frame = frame.drop(columns=cols_to_drop)
y = frame["ts15218"]
x = frame.drop(columns=["ts15218"])

ohec = OneHotEncoder(categories="auto")
x_ohec = ohec.fit_transform(x).todense()
model_count = 0
# dataset = make_dataset(x_ohec, y, n_splits=5)
# iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
# data_init_op = iterator.make_initializer(dataset)
# next_fold = iterator.get_next()


batch_sizes = [60, 120, 500]

hidden_layers = [
        (256, 1024, 2048, 512, 16),
        (2048, 512, 128, 16, 8),
        (16, 256, 512, 128, 8), 
        (512, 64, 8),
        (100)
]

optimizers = tf.train.AdamOptimizer
beta1s = [0.85, 0.9, 0.95],
beta2s = [0.95, 0.99]

lr = [0.001, 0.005, 0.01, 0.0005]
decay_coef = [0.6, 0.8, .95]

activations = [tf.nn.relu, tf.nn.tanh]
epoch_counts = [15, 25, 50, 100]


class Model:
    
    def __init__(
        self,
        x, y, n_splits,
        batch_size,
        layer_config,
        optimizer_params,
        epochs,
        activation,
        init_lr,
        decay_exp=None,
        path="./data/results/ffn/"
    ):
        global model_count
        model_count += 1
        self.model_no = model_count
        self.dir = path + f"model_{model_count:0>3}/"
        self.summary_dir = path + f"summaries/model_{model_count:0>3}/"
        shutil.rmtree(self.dir, ignore_errors=False)
        os.makedirs(self.dir)
        shutil.rmtree(self.summary_dir, ignore_errors=False)
        os.makedirs(self.summary_dir)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_splits = n_splits
        self.create(
            x, y, n_splits,
            batch_size,
            layer_config,
            optimizer_params,
            epochs,
            activation,
            init_lr,
            decay_exp=None
         )

        
    def create(
        self,
        x, y, n_splits,
        batch_size,
        layer_config,
        optimizer_params,
        epochs,
        activation,
        init_lr,
        decay_exp=None
        ):
        tf.reset_default_graph()

        self.g_step = tf.get_variable('global_step', trainable=False, initializer=0)

        self.input_data = tf.placeholder(tf.float32, [None, x.shape[1]])
        self.input_labels = tf.placeholder(tf.float32, [None])
        self.labels = self.input_labels - tf.constant(1, tf.float32)

        data = self.input_data
        for i, hidden_n in enumerate(layer_config):
            with tf.variable_scope(f"layer_{i}"):
                data = fully_connected(data, hidden_n, activation_function=activation)

        self.logits = tf.reshape(fully_connected(data, 1, activation_function=None), [-1])
        self.out = tf.round(tf.nn.sigmoid(self.logits))
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.loss = tf.reduce_mean(self.cross_entropy)

        if decay_exp is None:
            self.learning_rate = init_lr
        else:
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=init_lr,
                global_step=self.g_step,
                decay_steps=x.shape[0]//batch_size*epochs,
                decay_rate=decay_exp
            )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, **optimizer_params)
        self.step = self.optimizer.minimize(self.loss, global_step=self.g_step)

        self.accuracy, self.accuracy_op =  tf.metrics.accuracy(self.labels, self.out)
        self.precision, self.precision_op = tf.metrics.precision(self.labels, self.out)
        self.recall, self.recall_op = tf.metrics.recall(self.labels, self.out)
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy_op)
        tf.summary.scalar("precision", self.precision_op)
        tf.summary.scalar("recall", self.recall_op)
        tf.summary.scalar("f1", self.f1)
        
        self.summaries = tf.summary.merge_all()
    
    
    def run(self):
        train_writer = tf.summary.FileWriter(self.summary_dir + "/train", flush_secs=2)
        valid_writer = tf.summary.FileWriter(self.summary_dir + "/validation", flush_secs=2)
        saver = tf.train.Saver()
        
        batch_size = self.batch_size

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            for e in range(self.epochs):
                print(f"Starting epoch {e+1}...", end="\n", flush=True)
                for train_index, test_index in KFold(self.n_splits).split(self.x, self.y):
                    x_train, x_test = self.x[train_index], self.x[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]
                    

                    for i in range(x_train.shape[0] // batch_size):
                        batch = x_train[i*batch_size:(i+1)*batch_size]
                        labels = y_train[i*batch_size:(i+1)*batch_size]
                        # print(sess.run(self.labels, feed_dict={self.input_labels:labels}))
                        _, summaries = sess.run([self.step, self.summaries],
                                feed_dict={self.input_data: batch, self.input_labels: labels})
                        
                        gstep = sess.run(self.g_step)
                        train_writer.add_summary(summaries, gstep)
                    
                    loss, summ = sess.run([self.loss, self.summaries], feed_dict={self.input_data: x_test, self.input_labels: y_test})
                    gstep = sess.run(self.g_step)
                    valid_writer.add_summary(summ, gstep)
                    print("{0:<100}".format("|" + "." * ((100//batch_size)*i)), "|", sep="", end="\r", flush=True)
                    # print(f"|{'':100}|".format("."*((100//batch_size)*i)), sep="", end="\r", flush=True)
                    # print(f'{f"|{'':<{100//self.n_splits}}|":<100}', sep="", end="\r", flush=True)
                print(f"\nLatest loss: {loss}", flush=True)
                    
            save_path = saver.save(sess, os.path.join(self.dir, "model"), global_step=self.g_step)
            print(f"Saved model to {save_path}")



model = Model(
    x_ohec, y, 5,
    batch_size=120,
    # layer_config=[8192, 2048, 512, 128, 16],
    layer_config=[4096, 1024, 256, 32, 8],
    optimizer_params={"beta1":0.9, "beta2":0.99},
    epochs=25,
    activation=tf.nn.relu,
    init_lr=0.01,
    decay_exp=None,
    path="./data/results/ffn/"
)

model.run()

"""
for batch_size in batch_sizes:
    for layer_config in hidden_layers:
        for beta1 in beta1s:
            for beta2 in beta2s:
                optimizer_params = {"beta1":beta1, "beta2":beta2}
                for epochs in epoch_counts:
                    for activation in activations:
                        for init_lr in lr:
                            for decay_exp in decay_coef:
                                create_and_run(x_ohec, y, 3, batch_size, layer_config, optimizer_params, epochs, activation, init_lr, decay_exp)

                            create_and_run(x_ohec, y, 3, batch_size, layer_config, optimizer_params, epochs, activation, init_lr)
"""
