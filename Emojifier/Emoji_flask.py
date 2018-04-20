from flask import Flask, render_template, request,abort
import sys
import os
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
import keras.models
import re

from emo_utils import *
from pyemojify import emojify

from load import * 

app = Flask(__name__)

global model, graph

model, graph = init()

app = Flask(__name__)

app.config['DEBUG'] = True

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                                   
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               
        sentence_words = X[i].lower().split()        
        j = 0        
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1               
    return X_indices
      
@app.route('/index')
def write_message():
	return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def write():
    text = request.form['text']
    text = text.split()
    test_set = text.copy()
    text = np.array([' '.join(text)])
    
    for word in list(test_set):
        if word not in word_to_index:
            test_set.remove(word)  
    
    t = ' '.join(test_set)
    x_test = np.array([t])
    
    X_test_indices = sentences_to_indices(x_test, word_to_index, 10)       

    with graph.as_default():
        out = label_to_emoji(np.argmax(model.predict(X_test_indices)))
        out = text[0] +' '+ out
        
    
    return render_template('index.html', text = out)

if __name__ == "__main__":
    app.run(debug = True,use_reloader = False)
