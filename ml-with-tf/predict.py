"""Flower predictor

    author : Gaurav
    Usage :
        python predict.py ./test_images/orange_dahlia.jpg img_cls.h5 --top_k 2 --category_names label_map.json
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 64
IMG_SIZE = 224
LOCAL_MAPPING = 'label_map.json'
LOCAL_MODEL = 'img_cls.h5'

class_names = {}

def process_image(image_path):
    """process image
        
        Args:
            image_path : Image path
    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.squeeze(np.asarray(image))
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))/255.0
    return image


def predict(image_path, model, top_k): 
    """Predicts on provided images there top classes and preds
    """
    prediction = model.predict(np.expand_dims(process_image(image_path), axis=0))
    probs, labels =  tf.math.top_k(prediction, top_k)
    print("\nThese are the top Probs",probs.numpy()[0])
    top_classes = [class_names[str(label+1)] for label in labels.cpu().numpy()[0]]
    print(f"of top classes => {top_classes}")
    return probs.numpy()[0], top_classes



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('model')
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--category_names')
    
    args = parser.parse_args()  
    
    print(f"Params used => \nimage={args.image} \nmodel={args.model} \ntop_k={args.top_k} \ncategory_names={args.category_names}")
    # Read in the label_map file. 
    class_names = json.load(open(LOCAL_MAPPING if args.category_names is None else args.category_names)) 
    # Load model from path else use presaved 
    tf_model = tf.keras.models.load_model(args.model, compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
    top_k = 5 if args.top_k is None else args.top_k 
    
    probs, classes = predict(args.image, tf_model, top_k)