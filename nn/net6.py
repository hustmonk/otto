import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import site;
print site.getsitepackages()
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import *
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
#from nolearn.dbn import DBN
from lasagne.objectives import categorical_crossentropy
from adjustvariable import *
from matplotlib import pyplot
import nolearn
from common import *

import sys
import logging
import logging.config
logging.config.fileConfig("log.conf")
logger = logging.getLogger("example")
train_file = sys.argv[1]
test_file = sys.argv[2]
VALID = False
from classreduce import *
crs = ClassReduce()
def normal(X):
    Y = np.array(X)
    Y.clip(0, 10, Y)
    Y = np.log(Y+1.1)
    XY = []
    for i in range(X.shape[0]):
        xy = crs.expand(Y[i,:].tolist())
        XY.append(xy)
    
    X = np.array(XY)

    return X

if sys.argv[3] == 'True':
    VALID = True
out_dir = sys.argv[4]
def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels, ids = X[:, 1:-1].astype(np.float32), X[:, -1], X[:, 0].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    X = normal(X)
    scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    return X, y, ids,encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    if VALID:
        X, ids = X[:, 1:-1].astype(np.float32), X[:, 0].astype(str)
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = normal(X)
    #X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name=out_dir+'my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        #f.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))
X, y, ids, encoder, scaler = load_train_data(train_file)

X_test, ids_test = load_test_data(test_file, scaler)

num_classes = len(encoder.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 dropout0_p=0.2,
                 dense1_num_units=1500,
                 dropout1_p=0.3,
                 dense2_num_units=1500,
                 dropout2_p=0.2,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 objective=MyObjective,
                 #objective_loss_function=softmax,
                 objective_loss_function=categorical_crossentropy,
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=30)

"""
net0 = DBN(
    [num_features, 300, 300, num_classes],
    learn_rates=0.05,
    learn_rate_decays=0.9,
    l2_costs=0.005,
    epochs=100,
    verbose=1,
    dropouts=0.5
    )
"""
#parameters = dict(dropouts=[0.1,0.3,0.5,0.7])
#from nolearn.dataset import Dataset
from nolearn.grid_search import *
#dataset=Dataset(X, y)
#grid_search(dataset, net0, parameters, n_jobs=-1)
logger.info( net0 )
net0.fit(X, y)
print net0

#draw(net0)
#make_submission(net0, X, ids, encoder)
make_submission(net0, X_test, ids_test, encoder)
