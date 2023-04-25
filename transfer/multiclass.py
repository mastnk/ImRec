
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import glob
import numpy as np

# pip install numpy==1.19
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

from img2feat import CNN
cnn = CNN('vgg11')


dir_train = "data/train/"
dir_test = "data/test/"

labels = ["0", "1", "2", "3"]

def dataset(dir):
    X = None
    Y = None
    for i, label in enumerate(labels):
        d = os.path.join( dir, label )
        filenames = glob.glob( d+"/*.png" )

        imgs = []
        for filename in filenames:
            img = Image.open(filename).convert('RGB')
            imgs.append( np.array(img) )
        x = cnn( imgs )
        y = np.array( [1,]+[0]*(len(labels)-1) )
        y = np.roll( y, i )
        y = np.tile( y, (len(filenames), 1) )

        if( X is None ): X=x;
        else: X = np.concatenate( [X, x], 0 )
        if( Y is None ): Y=y;
        else: Y = np.concatenate( [Y, y], 0 )

    return X, Y

def softmax_proba( proba, alpha=20 ):
    proba = proba[:,:,1].transpose(1,0)
    ex = np.exp(proba*alpha)
    s = ex.sum(axis=1,keepdims=True)
    p = ex/s
    return p

def prediction( clf, X, alpha=20 ):
    Y_pred = clf.predict_proba(X)
    Y_pred = np.array(Y_pred)
    Y_pred = softmax_proba(Y_pred, alpha)
    return Y_pred

def acc( Y_true, Y_pred ):
    return ( np.argmax(Y_pred,axis=1) == np.argmax(Y_test,axis=1) ).astype(np.float).mean()

X_train, Y_train = dataset(dir_train)
X_test, Y_test = dataset(dir_test)

clf = MultiOutputClassifier( LogisticRegression() ).fit(X_train, Y_train)
Y_pred = prediction( clf, X_test )

print( Y_pred )
print( acc( Y_test, Y_pred ) )

