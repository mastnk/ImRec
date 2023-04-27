import argparse

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import pickle
import glob
import numpy as np

from sklearn.linear_model import LogisticRegression
from img2feat import CNN

###
def save_pkl( filename, clf ):
    with open( filename, mode='wb') as f:
        pickle.dump(clf,f,protocol=2)
###
def load_pkl( filename ):
    with open(filename, mode='rb') as f:
        clf = pickle.load(f)
    return clf

###
def load_images_labels( dirs, exts=["jpg", "png", "bmp"] ):
    exts = [e.lower() for e in exts] + [e.upper() for e in exts]

    images = []
    labels = []
    for i, dir in enumerate(dirs):
        filenames = []
        for ext in exts:
            filenames += sorted( glob.glob( dir+"/*.{}".format(ext) ) )

        for filename in filenames:
            images += [np.array(Image.open(filename).convert('RGB'))]
            labels += [i]
    return images, labels

###
class Dataset:
    def __init__( self, dir ):
        self.dir = dir

        self.dirs_train = sorted(glob.glob( os.path.join( self.dir, "train/*" ) ))
        self.dirs_test = sorted(glob.glob( os.path.join( self.dir, "test/*" ) ))

        for dir_train, dir_test in zip( self.dirs_train, self.dirs_test ):
            _, n_train = os.path.split(dir_train)
            _, n_test = os.path.split(dir_test)
            if( n_train != n_test ):
                raise ValueError("Folder names are unmatched: {} {}".format(n_train, n_test) )

    def data_train( self ):
        return load_images_labels( self.dirs_train )

    def data_test( self ):
        return load_images_labels( self.dirs_test )


###
class TenCrop:
    def __init__( self, border=16 ):
        self.border = border

    def __call__( self, images, labels=None ):
        images_aug = []

        for image in images:
            mirror = [image, np.fliplr( image )]
            for m in mirror:
                images_aug.append( m[self.border:-self.border, self.border:-self.border, :] )
                images_aug.append( m[self.border*2:, self.border:-self.border, :] )
                images_aug.append( m[:-self.border*2, self.border:-self.border, :] )
                images_aug.append( m[self.border:-self.border, self.border*2:, :] )
                images_aug.append( m[self.border:-self.border, :-self.border*2, :] )

        if( labels is None ):
            return images_aug
        else:
            labels_aug = []
            for label in labels:
                labels_aug += [ label for i in range(10)]
        return images_aug, labels_aug

###
def build_cnn( cnn ):
    try:
        return CNN(cnn)
    except:
        msg = "Can not build: {}\n".format( cnn )
        msg += "Please specify CNN name from:\n"
        for name in CNN.available_networks():
            msg += name + "\n"
        raise NotImplementedError( msg )

###
class Classifier:
    def __init__( self, cnn, clf=None, aug=None, C=1.0 ):
        self.fe = cnn
        if( clf is None ):
            self.cl = LogisticRegression(C=C,multi_class="multinomial",solver="saga",warm_start=True)
        else:
            self.cl = clf

        self.aug = aug

    def fit( self, images, y ):
        if( self.aug is not None ):
            images, y = self.aug( images, y )
        X = self.fe( images )

#        self.mean = X.mean(axis=0,keepdims=True)
#        X = X-self.mean

        self.scale = np.sqrt( np.square(X).sum(axis=1).mean() )
        X = X/self.scale

        y = np.array(y)
        self.cl.fit( X, y )

    def proba_pred( self, images ):
        if( self.aug is not None ):
            s0 = len( images )
            images = self.aug( images )
            s1 = len( images )
            n_aug = s1//s0

        X = self.fe( images )
#        X = X - self.mean
        X = X / self.scale

        proba = self.cl.predict_proba( X )

        if( self.aug is not None ):
            proba = proba.reshape( proba.shape[0]//n_aug, n_aug, proba.shape[1] ).mean(axis=1)

        return proba, np.argmax(proba, axis=1)

###
def acc( Y_true, Y_pred ):
    return ( Y_true == Y_pred ).astype(np.float).mean()

###
def main( images, cnn, tencrop, C  ):
    dataset = Dataset( images )
    X_train, y_train = dataset.data_train()
    X_test, y_test = dataset.data_test()

    if( tencrop > 0 ): aug = TenCrop( tencrop )
    else: aug = None

    clf = Classifier( cnn=build_cnn(cnn), aug=aug, C=C )
    clf.fit( X_train, y_train )
    save_pkl( "multiclass.pkl", clf )
    clf = load_pkl( "multiclass.pkl" )
    proba, pred = clf.proba_pred( X_test )

    for i in range(len(y_test)):
        cm = "OK" if pred[i] == y_test[i] else "NG"
        print( "{:.4f}, {:d}, {:d}, {:s}".format( proba[i,pred[i]], pred[i], y_test[i], cm ) )

    print( "Accuracy: {:.3f}".format( acc( y_test, pred ) ) )

###
if( __name__ == "__main__" ):
    parser = argparse.ArgumentParser(description="Multiclass image recognition sample")

    parser.add_argument("--images", default="images", help="It specifies the image folder.(images)" )
    parser.add_argument("--tencrop", type=int, default=0, help="It specifies the border of ten crop augmentation. 0: No augmentation.(0)")
    parser.add_argument("--cnn", default="vgg19", help="It specifies the network architecture.(vgg19)" )
    parser.add_argument("--C", type=float, default=1.0, help="It specifies the constraint of the logistic regression.(1.0)" )

    cfg = vars( parser.parse_args() )
    main( **cfg )

# % python multiclass.py --tencrop 16 --C 0.1

