import numpy as np
import keras.models
from keras.models import model_from_json
#from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from keras.models import load_model

from flask import Blueprint
load = Blueprint('load.py',__name__)


def init(): 
	loaded_model = load_model('emojify_model.hdf5')

	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	graph = tf.get_default_graph()

	return loaded_model,graph
