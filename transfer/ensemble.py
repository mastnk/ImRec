
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
#'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet', 'mobilenet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'vgg11', 'vgg13', 'vgg16', 'vgg19'
cnn = CNN('vgg19')


dir_train = "data/train/"
dir_test = "data/test/"
labels = ["0", "1", "2", "3"]

def TenCrop( imgs, border=16 ):
    aug = []

    for img in imgs:
        mirror = [img, np.fliplr( img )]
        for m in mirror:
            aug.append( m[border:-border, border:-border, :] )
            aug.append( m[border*2:, border:-border, :] )
            aug.append( m[:-border*2, border:-border, :] )
            aug.append( m[border:-border, border*2:, :] )
            aug.append( m[border:-border, :-border*2, :] )

    return aug

def dataset(dir, train=True, tencrop_border=16):
    X = None
    Y = None
    for i, label in enumerate(labels):
        d = os.path.join( dir, label )
        filenames = glob.glob( d+"/*.png" )

        imgs = []
        for filename in filenames:
            img = Image.open(filename).convert('RGB')
            imgs.append( np.array(img) )

        if( tencrop_border > 0 ):
            imgs = TenCrop( imgs, tencrop_border )
        x = cnn( imgs )

        y = np.array( [1,]+[0]*(len(labels)-1) )
        y = np.roll( y, i )
        if( train ): y = np.tile( y, (len(x), 1) )
        else: y = np.tile( y, (len(filenames), 1) )

        if( X is None ): X=x;
        else: X = np.concatenate( [X, x], 0 )
        if( Y is None ): Y=y;
        else: Y = np.concatenate( [Y, y], 0 )

    return X, Y


def softmax_proba( proba, proba_max=0.99999, n_aug=10 ):
    proba = np.clip( proba[:,:,1].transpose(1,0), 0, proba_max )
    logit = np.log( proba ) - np.log( 1-proba )

    if( n_aug > 1 ):
        logit = np.reshape( logit, [logit.shape[0]//n_aug, n_aug, logit.shape[1]] )
        logit = np.mean( logit, axis=1 )

    ex = np.exp(logit)
    s = ex.sum(axis=1,keepdims=True)
    p = ex/s
    return p

def prediction( clf, X, proba_max=0.99999, n_aug=10 ):
    Y_pred = clf.predict_proba(X)
    Y_pred = np.array(Y_pred)
    Y_pred = softmax_proba(Y_pred, proba_max, n_aug)
    return Y_pred

def acc( Y_true, Y_pred ):
    return ( np.argmax(Y_pred,axis=1) == np.argmax(Y_test,axis=1) ).astype(np.float).mean()

X_train, Y_train = dataset(dir_train, True,4)
X_test, Y_test = dataset(dir_test, False,4)

clf = MultiOutputClassifier( LogisticRegression() ).fit(X_train, Y_train)
Y_pred = prediction( clf, X_test )

Y_ind = np.argmax(Y_pred,axis=1)
Y_true = np.argmax(Y_test,axis=1)
for i in range(Y_test.shape[0]):
    cm = "OK" if Y_ind[i] == Y_true[i] else "NG"
    print( "{:.4f}, {:d}, {:d}, {:s}".format( Y_pred[i,Y_ind[i]], Y_ind[i], Y_true[i], cm ) )
print()
print( "Acc: {:.4f}".format( acc( Y_test, Y_pred ) ) )

