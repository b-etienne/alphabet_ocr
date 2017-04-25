# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:59:47 2017

@author: betienne
"""
import os
import tarfile
import numpy as np
import pickle
from scipy.misc import imread, imresize

# Download data from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
ROOT = os.getcwd()

def download():
    tar = tarfile.open(path_of_download)
    tar.extractall()
    tar.close()
    

def load_data(directory):
    filename = os.path.join(ROOT,'data\\data.pkl')
    
    if os.path.exists(filename):
        with open(filename,'rb') as f:
            X, y = pickle.load(f)
        return X, y    
        
    X = []
    y = []
    for folder in os.listdir(directory):
        if 10 < int(folder[-2:]) < 37:
            print("Processing {}".format(folder))
            X_b, y_b = load_batch(os.path.join(directory, folder))
            X.append(X_b)
            y.append(y_b)
    
    X, y = np.array(X).reshape((-1, 32, 32, 3)), np.array(y).reshape(-1)
    with open(filename,'wb') as f:
        pickle.dump((X, y), f)
        
    return X, y
    
    
def load_batch(folder):
    X = []
    y = []
    label = int(folder[-2:])-11
    for d in os.listdir(folder):
        img = imread(os.path.join(folder,d), mode='RGB')
        X.append(imresize(img, (32, 32, 3)))
        y.append([label])
    return np.array(np.reshape(X, (-1, 32, 32, 3))), np.array(y)
    
if __name__=='__main__':
    x = load_data(os.path.join(ROOT, 'data\\English\\Fnt')

     
    
        