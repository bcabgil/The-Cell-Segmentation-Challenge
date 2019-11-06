## Group 9001 Blanca Cabrera Gil & Laila Niazy

## Results
The best jaccard coefficient value obtained was 0.757. The used architecture was the U-Net network model using weighted loss function. And the parameter configuration the one shown in the following table:

| Base | Batch Size | Size | Epochs | Balanced Weights | Data Augmentation | Batch Normalization | Optimizer |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 64 | 2 | 128x128 | 5000 | True | True | False| RMSprop|

The next images show the set of predicted segmentation (left) againts its ground-truth mask (right). In the predicted image the values are thresholded to as values greater than 0 are equal to 1.

![](https://github.com/bcabgil/The-Cell-Segmentation-Challenge/blob/master/Gifs/predicted_last.gif)	![](https://github.com/bcabgil/The-Cell-Segmentation-Challenge/blob/master/Gifs/gif_gt_mask_prediction.gif)

## How the repo is organized:

-For all four models, we have json files that can be used to change all the parameters. The json files can be found in the folder "Json-Files".

-In the folder "PhC-C2DH-U373", we saved the training and test data with the ground truth.

-In the folder "Tested Model", we have the code for training all the four models.

-In the folder "Saved Model", we have all the models we trained, saved as an '.h5'-File

-In the folder "Plots", we have four folders, where the plots for each trained models can be found.

-In the folder "Utils", the different functions we use in order to train the model can be found such as the data loader (to read the data), the plotter (to plot the loss curve), the model itself etc.

## How to execute the Main.py:

-The python file "Main.py" can be executed in the terminal then one can choose which model should be trained:

    If number 1 is typed, then the model U-Net is used.
    
    If number 2 is typed, then the model U-Net with LSTM is used.
    
    If number 3 is typed, then the model U-Net with the weighted loss function is used.
    
    If number 4 is typed, then the model U-Net with LSTM and the weighted loss function is used.
    
-If one wants to change the parameter configuration, then the Json-Files in the folder can change accordingly.
    
##The orginal configuration, we tested the four models with are:
- base = 32
- batch_size = 2
- LR = 0.001
- SDRate = 0.5
- batch_normalization = True
- spatial_dropout = True
- epochs = 1000
- final_neurons= 1 #binary classification
- final_afun = "sigmoid" #activation function
- weight_strength = 1.
- With Data augmentation
- Accuracy = jaccard coefficient
- Loss = binary crossentrophy for non weighted unet/ weighted binary accuracy for weighted unet
- We didn't use balanced weights to encounter the class imbalanced problem
- Optimizer = Adam


