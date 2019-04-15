import pandas as pd
import tensorflow as tf
import keras as k
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dot
from keras.models import Model, model_from_json
from keras.regularizers import l2
from keras.constraints import nonneg
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.optimizers import Adam
from time import time
from datetime import datetime
from IPython.display import SVG
import warnings
from os import listdir
from os.path import isfile, join
import re



def supervised_model(n, layers=[80, 40, 20]):
    """
    Feed in numer of features
    Return a model for embedding training
    """
    
    # define input placeholders
    left_input = Input((n,), name='LeftInput')
    right_input = Input((n,), name='RightInput')
      
    # define shared layers
    i = 0
    left_X = left_input
    right_X = right_input
    for l in layers:
        i += 1
        #shared_layer = Dense(l, kernel_regularizer=l2(0.0001), activation='relu', name='DenseLayer'+str(i))
        shared_layer = Dense(l, activation='relu', name='DenseLayer'+str(i))
        left_X = shared_layer(left_X)
        right_X = shared_layer(right_X)
    
    # define cosine similarity layer
    X = Dot(1, normalize=True, name='CosineSimilarity')([left_X, right_X])
    
    # define output layer and force the weights to be non-negative to help with convergence
    X = Dense(1, kernel_constraint=nonneg(), name='Output')(X)
    
    # create model
    model = Model(inputs=[left_input, right_input], outputs=X, name='LearningEmbeddings')

    return model



def scoring_model(model):
    """
    Feed in full supervised model
    Return a model for retrieving embeddings
    """
    
    scoring_model = k.models.Model(inputs=model.input[0], name='ScoringEmbeddings',
                                   outputs=model.get_layer('DenseLayer3').get_output_at(0))
    for layer in scoring_model.layers:
        layer.trainable = False
        
    return scoring_model



def save_model(model, name):
    """
    Feed in model and its name
    Save on disk
    """

    ds = datetime.fromtimestamp(time()).strftime('%Y-%m-%d')
    
    # serialize model to JSON
    with open(name+' '+ds+'.json', 'w') as json_file:
        json_file.write(model.to_json())

    # serialize weights to HDF5
    model.save_weights(name+' '+ds+'.h5')

    

def load_model(name):
    """
    Feed in model name and datestamp
    Load from disk and return model
    """
    
    # load JSON and create model
    with  open(name+'.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(name+'.h5')
    print("Model '"+name+"' was loaded successfully")
    
    return model



def visualise_model(model, name):
    """
    Feed in model and its name
    Visualise the model and save on disk
    """

    ds = datetime.fromtimestamp(time()).strftime('%Y-%m-%d')
    vis = SVG(model_to_dot(model).create(prog='dot', format='svg'))
    plot_model(model, to_file=name+' Plot '+ds+'.png', show_shapes=True, show_layer_names=True)
    return vis



def find_latest_model(name):
    """
    Feed in the name of the model
    Return model object (.json) and model weights (.h5)
    """
    
    # get list of files in the working folder
    files = [f for f in listdir(".") if isfile(join(".", f))]
    
    # find names of model objects and extract names
    name_regex = re.compile(name+'.*json')
    files = list(filter(name_regex.search, files))
    date_regex = re.compile(r'(\d\d\d\d-\d\d-\d\d)')
    dates = date_regex.findall(str(files))
    
    # get the latest datestamp
    if len(dates) > 0:
        ds = max(list(map(lambda d: datetime.strptime(d, '%Y-%m-%d'), dates))).strftime('%Y-%m-%d')
        model = load_model(name+' '+ds)
    else:
        return "No file found"
    
    return model



def get_model(load_model=True, model_name='Learning Embeddings', columns=['TRating', 'GP1_Derived_Soft Diff'], 
              epochs=1, full_model=None, data=None):
    """
    Feed in model name and parameters. Specify whether model should be rebuilt from scratch or loaded from disk.
    If model needs to be retrained, feed in data. If scoring model needs to be constructed, feed in learning model.
    Returns model plot and model object
    """
    
    if load_model == True:
        model = find_latest_model(model_name)
        if model != "No file found":
            model.summary()
            plot = visualise_model(model, model_name)
            return plot, model
        else:
            print('The requested model was not found on disk. Rebuilding the model')
        
    if model_name == 'Learning Embeddings':
        if data is None:
            warnings.warn('Please provide data')
            return [None, None]
        else:
       
            # build the model
            X_left, X_right, Y = data
            n = X_left.loc[:,columns[0]:columns[1]].shape[1]
            model = supervised_model(n) 
            model.summary()
            plot = visualise_model(model, model_name)
        
            # train the model
            print("Training the model")
            adam = k.optimizers.Adam(lr=0.0001)
            model.compile(optimizer = adam, loss = "mse")
            model.fit(x=[X_left.loc[:,columns[0]:columns[1]], X_right.loc[:,columns[0]:columns[1]]], y=Y/Y.std(), 
                      epochs=epochs, batch_size=128, verbose = 2,
                      callbacks=[TQDMNotebookCallback(), EarlyStopping(min_delta=0, patience=3)])
            save_model(model, model_name)
            return plot, model
        
    elif model_name == 'Scoring Embeddings':
        if full_model is None:
            warnings.warn('Please provide a model object')
            return [None, None]
        else:
            model = scoring_model(full_model)
            model.summary()
            plot = visualise_model(model, model_name)
            return plot, model
    
    else:
        warnings.warn('This model is unknown')
        return [None, None]

    
    
def model_predict(model, data, columns=['TRating', 'GP1_Derived_Soft Diff']):
    """
    Feed in model object and data.
    Returns ...
    """
    
    if data is None:
            warnings.warn('Please provide data')
            return
    
    if model.name == 'LearningEmbeddings':
        if len(data) < 2:
            warnings.warn('Please provide data in the format [X_left, X_right]')
            return
        X_left, X_right = data[0:2]
        pred = model.predict([X_left.loc[:,columns[0]:columns[1]], X_right.loc[:,columns[0]:columns[1]]])
        return pd.DataFrame(pred)
    
    elif model.name == 'ScoringEmbeddings':
        X_left = data[0]
        pred = model.predict([X_left.loc[:,columns[0]:columns[1]]])
        return pd.DataFrame(pred)
        
    else:
        warnings.warn('This model is unknown')
        return