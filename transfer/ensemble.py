
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import pickle
import glob
import numpy as np

from sklearn.linear_model import LogisticRegression

from img2feat import CNN
#'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet', 'mobilenet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'vgg11', 'vgg13', 'vgg16', 'vgg19'
cnn = CNN('vgg19')


dir_train = "data/train/"
dir_test = "data/test/"
labels = ["0", "1", "2", "3"]

def save_clf( filename, clf ):
    with open( filename, mode='wb') as f:
        pickle.dump(clf,f,protocol=2)

def load_clf( filename ):
    with open(filename, mode='rb') as f:
        clf = pickle.load(f)
    return clf

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
    Y = []
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
        if( X is None ): X=x;
        else: X = np.concatenate( [X, x], 0 )

        if( train ): n = len(x)
        else: n = len(filenames)
        Y += [i for _ in range(n) ]

    return X, np.array(Y)

def acc( Y_true, Y_pred ):
    return ( Y_true == Y_pred ).astype(np.float).mean()

def ensemble( Y_proba, n_aug ):
    return Y_proba.reshape( Y_proba.shape[0]//n_aug, n_aug, Y_proba.shape[1] ).mean(axis=1)

X_train, Y_train = dataset(dir_train, True,4)
X_test, Y_test = dataset(dir_test, False,4)

clf = LogisticRegression(C=10,multi_class="multinomial",solver="newton-cg",warm_start=True).fit(X_train, Y_train)
save_clf( "ensemble.pkl", clf )
clf = load_clf( "ensemble.pkl" )

Y_proba = clf.predict_proba(X_test)
Y_proba = ensemble( Y_proba, 10 )
Y_pred = np.argmax(Y_proba,axis=1)

for i in range(Y_test.shape[0]):
    cm = "OK" if Y_pred[i] == Y_test[i] else "NG"
    print( "{:.4f}, {:d}, {:d}, {:s}".format( Y_proba[i,Y_pred[i]], Y_pred[i], Y_test[i], cm ) )
print()
print( "Acc: {:.4f}".format( acc( Y_test, Y_pred ) ) )
