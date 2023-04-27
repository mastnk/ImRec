import argparse

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import os
import glob
import numpy as np

def load_images( dir, exts=["jpg", "png", "bmp"] ):
    exts = [e.lower() for e in exts] + [e.upper() for e in exts]

    filenames = []
    for ext in exts:
        filenames += sorted( glob.glob( dir+"/*.{}".format(ext) ) )

    images = []
    for filename in filenames:
        images += [Image.open(filename).convert('RGB')]
    return filenames, images

def saliencymap( model, image ):
    model.eval()
    image_tensor = transforms.ToTensor()( image )
    image_tensor = torch.unsqueeze(image_tensor, 0)
    input_tensor = image_tensor.requires_grad_()
    output_tensor = model(input_tensor)
    grads = torch.autograd.grad(outputs=output_tensor, inputs=input_tensor,
                                grad_outputs=torch.ones(output_tensor.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.detach()
    grads = grads[0]
    grads = grads.square().sum(axis=0).sqrt()
    grads = grads / grads.max()
    return grads.numpy()

def save_with_colormap( filename, array, colormap="jet" ):
    cmap = plt.get_cmap(colormap)
    array = cmap(array, bytes=True)
    im = Image.fromarray(array)
    im.save(filename)

def main( inputs, outputs, outputs_saliency, size ):
    # Load a pre-trained model
    model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)

    os.makedirs( outputs, exist_ok=True)
    os.makedirs( outputs_saliency, exist_ok=True)

    filenames, images = load_images( inputs )

    for filename, image in zip(filenames,images):
        title = os.path.splitext( os.path.split( filename )[1] )[0]
        print( title )
        s = image.size
        if( s[0] >= s[1] ): #width>=height
            a = float(size)/float(s[1])
            s = [int(s[0]*a),size]
        else: #width<height
            a = float(size)/float(s[0])
            s = [size, int(s[1]*a)]

        image = image.resize(s, Image.LANCZOS)

        sal = saliencymap( model, image )
        save_with_colormap( os.path.join( outputs_saliency, title+".png" ), sal )

        if( s[0] < s[1] ): #width<height
            image = image.transpose(Image.FLIP_ROTATE_90)
            sal = sal.transpose(1,0)

        x = sal.mean( axis=0 )
        x = np.convolve(x, np.ones(size), mode='valid')
        p = x.argmax()

        left = p
        right = p+size
        top = 0
        bottom=size
        image = image.crop((left, top, right, bottom))

        if( s[0] < s[1] ): #width<height
            image = image.transpose(Image.FLIP_ROTATE_270)
            sal = sal.transpose(1,0)

        image.save( os.path.join( outputs, title+".png" ) )


###
if( __name__ == "__main__" ):
    parser = argparse.ArgumentParser(description="Automatic image cropping")

    parser.add_argument("--inputs", default="inputs")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--outputs_saliency", default="outputs_saliency")
    parser.add_argument("--size", type=int, default=256)

    cfg = vars( parser.parse_args() )
    main( **cfg )
