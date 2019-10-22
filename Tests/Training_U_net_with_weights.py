from Utils.Data_Loader import get_train_test_data
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from Utils.plotter import plotter
from Utils.losses import *
from Utils.u_net import u_net
from Utils.Data_Augmentation import*
import sys
import math
import json
from sklearn.utils import class_weight
import numpy as np

def training_unet_with_weights():
    
    #Data parameters
    img_ch = 1
    fold1 = '01'
    fold2 = '01_GT/SEG'
    data_path = 'PhC-C2DH-U373'

    
    #Import different configurations for the model
    with open('Json-Files/initialize_unet_weights_config.json', 'r') as f:
        distros_dict = json.load(f)
        
    for i,distro in enumerate(distros_dict): 
    
        #Network parameters
        weight_strength = 1.
        min_loss = []
        min_jaccard = []

        #Get the train and test images, masks & weight maps
        x_train, x_val, y_train, y_val, weight_train, weight_val = get_train_test_data(fold1, fold2, data_path, distro['img_h'], distro['img_w'])

        #Create the train generator for data augmentation
        train_generator = generator_with_weights(x_train, y_train, weight_train, distro['batch_size'])

        #Create the validation generator
        val_generator = generator_with_weights(x_val, y_val, weight_val, distro['batch_size'])

        #######Compile the u-net model with the previously stated parameters#######

        model, input_weights = u_net(distro['base'], distro['img_w'], distro['img_h'], img_ch, distro['batch_normalization'], distro['SDRate'], distro['spatial_dropout'], distro['number_labels'],distro['activation_function'],lstm = False, weighted =True)

        #Compile the model with weighted cross-entropy loss and metric as jaccard_distance
        model.compile(optimizer = Adadelta(lr=0.001), loss = weighted_bce_loss(input_weights, weight_strength), metrics =[jaccard_acc])


        #######Fit the data into the model#######

        #If the configuration contains balanced weights
        if distro['balanced_weights']:
             # Calculate the weights for each class so that we can balance the data
            y = y_train.reshape(y_train.shape[0],y_train.shape[1]*y_train.shape[2]).shape
            weights = class_weight.compute_sample_weight('balanced',y)
        else:
            weights = "auto"

        History = model.fit_generator(train_generator, epochs=distro['epochs'], verbose=1, max_queue_size=1, validation_steps=len(x_val), validation_data=([x_val, weight_val], y_val), shuffle=False, class_weight=weights, steps_per_epoch = math.ceil(len(x_train) / distro['batch_size']))


        # Save the weights
        model.save_weights('Models/U_Net_with_Weights/U_Net_Weights_Adadelta_Base_{}.h5'.format(distro['base']))
        #Plots per fold
        fig_loss, fig_dice, min_loss_arg, min_jaccard_arg = plotter(History)
        fig_loss.savefig('Plots/U_Net_with_Weights/Learning_Curve_U_Net_with_Weights_Adadelta_Base_{}.png'.format(distro['base']))
        fig_dice.savefig('Plots/U_Net_with_Weights/Jaccard_Score_Curve_U_Net_with_Weights_Adadelta_Base_{}.png'.format(distro['base']))

        min_loss.append(min_loss_arg)
        min_jaccard.append(min_jaccard_arg)
        model_number = np.argmax(min_jaccard)
        
       
    print("The model that obtained the best scores is the one from fold {}, with loss: {}, and jaccard: {}".format(model_number, min_loss[model_number], min_jaccard[model_number]))

