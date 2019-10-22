from Utils.Data_Loader import get_train_test_data
from tensorflow.keras.optimizers import Adam
from Utils.plotter import plotter
from Utils.losses import *
from Utils.u_net import u_net
from Utils.Data_Augmentation import DataAugmentation
import sys
import numpy as np
from sklearn.utils import class_weight
import json

def training_u_net():
    
    #########Initialize the data for the images#############
    img_ch = 1
    fold1 = '01' #folder with training data
    fold2 = '01_GT/SEG' #folder with ground truth
    data_path = 'PhC-C2DH-U373'
    min_loss = []
    min_jaccard = []
    
    #########Initialize the data for the augmentation#############
    rotation_range = 10
    width_shift = 0.1
    height_shift = 0.1
    rescale = 0.2
    horizontal_flip = True

    #########Import the different configurations for the model#############
    with open('Json-Files/intialize_unet.json', 'r') as f:
        distros_dict = json.load(f)
        
    for distro in distros_dict: 

        #########Data Augmentation#############
        train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift,rescale,horizontal_flip)
        #########Get the images and their masks (labels)#############
        x_train, x_val, y_train, y_val, _ , _ = get_train_test_data(fold1, fold2, data_path, distro['img_h'], distro['img_w'])
      

        #########Initialize the model and compile it #############
        model= u_net(distro['base'],distro['img_h'], distro['img_w'], img_ch, distro['batch_normalization'], distro['SDRate'], distro['spatial_dropout'], distro['number_of_labels'],distro['activation_function'], lstm = False, weighted =False)
        model.compile(optimizer = Adam(lr=distro['LR']), loss = distro['loss_function'], metrics =[jaccard_acc])

        ###############choosing between using weights for unbalanced data or uniform weights###############
        if distro['balanced_weights']:
            # Calculate the weights for each class so that we can balance the data
            y = y_train.reshape(y_train.shape[0],y_train.shape[1]*y_train.shape[2]).shape
            weights = class_weight.compute_sample_weight('balanced',y)
        else: 
            weights = 'auto'

        #choose between using data augmentation or not
        if distro['data_augmentation']:
            # Add the class weights to the training and fit the data into the model 
            History = model.fit_generator(train_datagen.flow(x_train, y_train,batch_size = distro['batch_size']), validation_data = val_datagen.flow(x_val,y_val), epochs = distro['epochs'], verbose = 1, class_weight=weights)   
        else:
            # Add the class weights to the training and fit the data into the model 
            History = model.fit(x_train, y_train, epochs=distro['epochs'], batch_size=distro['batch_size'], class_weight=weights,verbose = 1, validation_data =(x_val,y_val))


        #################plotting###########################################

        fig_loss, fig_acc, min_loss_arg, min_jaccard_arg  = plotter(History)

        fig_loss.savefig('Plots/U-Net/Learning_curve_BN_{}_DA_{}_BW_{}.png'.format(distro['batch_normalization'],distro['data_augmentation'],distro['balanced_weights']))
        fig_acc.savefig('Plots/U-Net/Jaccard_Loss_Curve_BN_{}_DA_{}_BW_{}.png'.format(distro['batch_normalization'],distro['data_augmentation'],distro['balanced_weights']))

        ################saving the model to the folder Models##########################
        # Save the weights
        model.save_weights('Models/U_Net/U_Net_weights_BN_{}_DA_{}_BW_{}.h5'.format(distro['batch_normalization'],distro['data_augmentation'],distro['balanced_weights']))
        #Get the best score and model
        min_loss.append(min_loss_arg)
        min_jaccard.append(min_jaccard_arg)
        model_number = np.argmax(min_jaccard)


    print("The model that obtained the best scores is the one from fold {}, with loss: {}, and jaccard{}".format(model_number, min_loss[model_number], min_jaccard[model_number]))