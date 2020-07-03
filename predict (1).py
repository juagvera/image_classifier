import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow.keras.layers
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pdb
import scipy
import argparse





def Main():
    class_name= 'label_map.json'
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="the image file", type=str)
    parser.add_argument("modelname", help="the modelo file", type=str)
    parser.add_argument("--top_k", help="Top quantity of probabilites", required=False, type=int)
    parser.add_argument("--category_names", help="Json file", required=False, type=str)
    
    
    
    args = parser.parse_args()
    
    if args.top_k:
       k = args.top_k
    else:
        k = 3
        
    if args.category_names:
        jason_file_name = args.category_names
    else:
        jason_file_name = 'label_map.json'
    
    image = args.filename
    saved_model_path = args.modelname
    
    dict = [image, saved_model_path, k, class_name]
    return dict



command_line = Main()

#The image file
image= command_line[0]

#The model file
saved_model_path = command_line[1]

#the top probabilities
k= command_line[2]

#The json  file
jason_file_name = command_line[3]


def process_image(image):
    global dsize
    image_size=224
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image,(image_size, image_size) )
    image/=255
    return image

def predict(image_path=None, model=None, top_k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    print(image_path.split("/")[-1])
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs=model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

with open(jason_file_name, 'r') as f:
    class_names = json.load(f)

#image='./test_images/orange_dahlia.jpg'


model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer':hub.KerasLayer})

prediction, classes = predict(image_path=image,model=model,top_k=int(k))


pred=prediction.numpy().squeeze().tolist()
class_pred=classes.numpy().squeeze().tolist()

print(f'\n\n The {image} is a {class_names.get(str(class_pred[int(np.where(np.amax(pred))[0])]))}\n\n')

cont=0
for i in class_pred:
    pred_f=round(pred[cont],2)
    print(f'The {class_names.get(str(i))}: {pred_f}')
    cont+=1


