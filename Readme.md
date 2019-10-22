## Group 9001 Blanca Cabrera Gil & Laila Niazy
## How the repo is organized:

-For all four models, we have json files that can be used to change all the parameters. The json files can be found in the folder "Json-Files".

-In the folder "PhC-C2DH-U373", we saved the training and test data with the ground truth.

-In the folder "Tests", we saved all the four models that we trained.

-In the folder "Plots", we have four folders, where the plots for each trained models can be found.

-In the folder "Utils", the different functions we use in order to train the model can be found such as the data loader (to read the data), the plotter (to plot the loss curve), the model itself etc.

-The python file "Main.py" can be executed in the terminal then one can choose which model should be trained:
    If number 1 is typed, then the model U-Net is used.
    If number 2 is typed, then the model U-Net with LSTM is used.
    If number 3 is typed, then the model U-Net with the weighted loss function is used.
    If number 4 is typed, then the model U-Net with LSTM and the weighted loss function is used.
    
##The orginal configuration, we tested the four models with are:
- base = 32
- batch_size = 2
- LR = 0.001
- SDRate = 0.5
- batch_normalization = False
- spatial_dropout = True
- epochs = 100
- final_neurons= 1 #binary classification
- final_afun = "sigmoid" #activation function
- weight_strength = 1.
- With Data augmentation
- Accuracy = jaccard coefficient
- Loss = binary crossentrophy for non weighted unet/ weighted binary accuracy for weighted unet
- Used balanced weights to encounter the class imbalanced problem
- Optimizer = Adam


