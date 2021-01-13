import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

import time
import os
import shutil

import copy
import datetime
from successive_halving import *
from layers import ConvNet
import network_operators
import utils
from hyperparameters import *

if __name__ == '__main__':
    """
    Main method. It runs n_experiments and saves the results 
    Args:
    Returns:
    """

    # Load data
    trainloader, validloader, testloader = utils.prepare_data(batch_size=128, valid_frac=0.1)

    # Get the first batch of the training set. We use this batch for calculating
    # batch normalization parameters and when we need to initializa other parameters.
    
    inputs, targets = trainloader[0]
    batch, batch_y = Variable(inputs.cuda()), Variable(targets.cuda())

    # Create and train the vanilla model.
    vanilla_model = create_vanilla(trainloader, expfolder)

    # Start experiments
    for experiment_num in range(0, n_experiments):

        start_run = datetime.datetime.now()
        
        # Prepare output folders for this experiment
        folder_out = expfolder + 'run_' + str(experiment_num) + '/'
        os.mkdir(folder_out)

        # Load the inital model parameters
        initial_model = vanilla_model
        initial_model['pytorch_model'].load_state_dict(torch.load(expfolder + "vanilla_model"), strict=False)

        sh_idx = 0 # Successive halving run index
        is_continue_mutating = True

        while is_continue_mutating:

            # Create a folder for all models of this experiment, using successive halving iteration index
            savepath = folder_out + str(sh_idx) + '/'
            os.mkdir(savepath)

            st = time.time()
            
            # Given the initial model, start the process.
            # Creates the population, trains and selects the best one  
            initial_model = get_winner_child_network(n_models, batch, n_mutations, trainloader, validloader, n_epochs_total, initial_model, savepath, folder_out)
            end = time.time()

            # Check the number of parameters
            pytorch_total_params = sum(p.numel() for p in initial_model['pytorch_model'].parameters() if p.requires_grad)

            # Quit mutating if we reached the threshold 
            if pytorch_total_params > max_n_params:
                is_continue_training = False

            # Quit if the experiments are taking more than 23 hours
            if datetime.datetime.now() > (start_run + datetime.timedelta(hours=mutation_time_limit_hours)):
                is_continue_mutating = False
            sh_idx += 1

        # Final training of the winner model
        final_training(initial_model, folder_out)
           

