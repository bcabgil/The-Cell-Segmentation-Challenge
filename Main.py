from Tests.Training_U_net import training_u_net
from Tests.Training_U_net_with_LSTM import training_u_net_lstm
from Tests.Training_U_net_with_weights import training_unet_with_weights
from Tests.Weighted_LSTM_Unet import weighted_lstm_unet

def main(argv):
   
    if argv == '1':
        print('doing training_u_net')
        history_1 = training_u_net()
    elif argv == '2':
        print('doing training_u_net_lstm')
        history_2 = training_u_net_lstm() 
    elif argv == '3':
        print('doing training_unet_with_weights')
        history_3 = training_unet_with_weights()
        
    elif argv == '4':
        print('doing weighted_u_net_lstm')
        history_4 = weighted_lstm_unet()
        
    else:
        print('wrong task number')


if __name__ == "__main__":
    #input in the console is the number of the task
    task = input("Enter the number of test to perform: ")
    main(task)
    

