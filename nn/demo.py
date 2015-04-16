import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import sys
train_file = sys.argv[1]
test_file = sys.argv[2]
VALID = False
def normal(X):
    Y = np.array(X)
    X.clip(0, 1, X)
    Y.clip(0, 10, Y)
    Y = np.log(Y+1)
    XY = []
    for i in range(X.shape[0]):
        xy = X[i,:].tolist() + Y[i,:].tolist()
        XY.append(xy)
    
    X = np.array(XY)

    print X[0]
    return X

if sys.argv[3] == 'True':
    VALID = True
out_dir = sys.argv[4]
def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    X = normal(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    if VALID:
        X, ids = X[:, 1:-1].astype(np.float32), X[:, 0].astype(str)
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = normal(X)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name=out_dir+'my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))
X, y, encoder, scaler = load_train_data(train_file)

X_test, ids = load_test_data(test_file, scaler)

num_classes = len(encoder.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=200,
                 dropout_p=0.5,
                 dense1_num_units=200,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=100)

net0.fit(X, y)

make_submission(net0, X_test, ids, encoder)
