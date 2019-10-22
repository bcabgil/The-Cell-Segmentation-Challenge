# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:37:45 2019

"""

##plotter
import matplotlib.pyplot as plt
import numpy as np

def plotter(History):
    #Plotting the training and validation learning loss curve 
    fig_loss = plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend(); 

    
    #Plotting the jaccard distance for the training and test set
    fig_acc = plt.figure(figsize=(4,4))
    plt.title("Jaccard Accuracy Learning Curve")
    plt.plot(History.history["jaccard_acc"], label="jaccard_coeff")
    plt.plot(History.history["val_jaccard_acc"], label="val_jaccard_coeff")
    plt.plot(np.argmax(History.history["val_jaccard_acc"]),
             np.max(History.history["val_jaccard_acc"]),
             marker="x", color="b", label="best model")
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Coefficient')
    plt.legend();


    return fig_loss, fig_acc, np.min(History.history["val_loss"]), np.max(History.history["val_jaccard_acc"])


