import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from layers import ConvNet

import network_operators
import utils
from hyperparameters import *

import os
import shutil

import copy
import datetime


def create_vanilla(trainloader, expfolder):
    """
    Function to create the vanilla network

    Args:
        trainloader: trainloader object from Pytorch
        expfolder: folder to save the model
    Returns:
        Trained vanilla model
    """
    # Layers
    layer0 = {'type': 'input', 'params': {'shape': (32,32,3)}, 'input': [-1],'id': 0}
    layer1 = {'type': 'conv', 'params': {'channels': 64, 'ks1': 3, 'ks2': 3, "in_channels": 3}, 'input': [0], 'id': 1}
    layer1_1 = {'type': 'batchnorm', 'params': {"in_channels": 64}, 'input': [1], 'id': 2}
    layer1_2 = {'type': 'activation', 'params': {}, 'input': [2], 'id': 3}
    layer4 = {'type': 'pool', 'params': {'pooltype': 'max', 'poolsize': 2}, 'input': [3],'id': 10}
    layer5 = {'type': 'conv', 'params': {'channels': 128, 'ks1': 3, 'ks2': 3, "in_channels": 64}, 'input': [10], 'id': 11}
    layer5_1 = {'type': 'batchnorm', 'params': {"in_channels": 128}, 'input': [11], 'id': 12}
    layer5_2 = {'type': 'activation', 'params': {}, 'input' : [12], 'id': 13}
    layer8 = {'type': 'pool', 'params': {'pooltype': 'max', 'poolsize': 2}, 'input': [13],'id': 20}
    layer9 = {'type': 'conv', 'params': {'channels': 256, 'ks1': 3, 'ks2': 3, "in_channels": 128}, 'input': [20], 'id': 21}
    layer9_1 = {'type': 'batchnorm', 'params': {"in_channels": 256}, 'input': [21], 'id': 22}
    layer9_2 = {'type': 'activation', 'params': {}, 'input' : [22], 'id': 23}
    layer11 = {'type': 'dense', 'params': {'units': 10, "in_channels": 256, "in_size": 8}, 'input': [23], 'id': 27}

    # Hyperparameters, optimizer, lr scheduler, loss
    lr_vanilla = 0.01
    opt_algo = {'name': optim.SGD, 'lr': lr_vanilla, 'momentum': 0.9, 'weight_decay': 0.0005, 'alpha': 1.0}
    sch_algo = {'name': optim.lr_scheduler.CosineAnnealingLR, 'T_max': 5, 'eta_min': 0, 'last_epoch': -1}
    comp = {'optimizer': opt_algo, 'scheduler': sch_algo, 'loss': nn.CrossEntropyLoss, 'metrics': ['accuracy']}

    # We keep a model desriptor dictionary to reach the layers and other components of the model
    # in O(1) time 
    model_descriptor = {}
    model_descriptor['layers'] = [layer0, layer1, layer1_1, layer1_2,
                                layer4, layer5, layer5_1, layer5_2,
                                layer8, layer9, layer9_1, layer9_2, layer11]

    model_descriptor['compile']= comp

    # Generate the model
    mod = ConvNet(model_descriptor)
    mod.cuda()

    vanilla_model = {'pytorch_model': mod, 'model_descriptor': model_descriptor, 'topo_ordering': mod.topo_ordering}


    # Train the vanilla model
    vanilla_model['pytorch_model'].fit(trainloader, train_type = 'vanilla', epochs=20)

    # Save its parameters for further use
    torch.save(vanilla_model['pytorch_model'].state_dict(), expfolder + "vanilla_model")

    return vanilla_model

def get_sh_epochs(n_epochs_total,n_models,n_minibatches):
    """
    Function to calculate epochs in terms of successive halving budget to be used in the scheduler

    Args:
        n_epochs_total: training budget for the first iteration of successive halving
        n_models: number of child networks to generate
        n_minibatches: number of minibatches in train set
    Returns:
        Returns successive halving budget
    """
    
    sch_epochs = 0

    for s in range(0, int(np.log2(n_models)) + 1):
        sch_epochs += n_epochs_total * (2**(s))
  
    return sch_epochs * n_minibatches

def mutate_network(n_mutations, model, model_idx, batch, max_n_params, times):
    """
    Function to mutate the initial network n_mutations times

    Args:
        n_mutations: number of mutations to apply
        model: initial model to mutate
        model_idx: index of the model
        batch: first minibatch of the training set
        max_n_params: number of parameters threshold
        times: a list to keep track of mutating time, training time and overall processing time
    Returns:
        Returns the mutated model and updated times 
    """
    
    for _ in range(0, n_mutations):

        time_mut_s = time.time()

        # Do not mutate the first child. The mutations might make it worse, so keep it 
        # If it is the winner of the given round, it might be mutated later.
        if model_idx != 0:
            
            # Given a model, mutate and return the new model
            model = network_operators.MutateNetwork(model, batch, mutations_probs)

            time_mut_e = time.time()

            times[1] = times[1] + (time_mut_e - time_mut_s)

            # Check the number of parameters reached. If it is more than the threshold stop mutation
            pytorch_total_params = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)

            if pytorch_total_params > max_n_params:
                break
    return times, model
  
def generate_population_and_train(n_models, batch, initial_model, init_weights_path, sch_epochs, descriptors, trainloader, n_epochs_each, performance, validloader, folder_out, savepath):
    """
    Function to generate a population and train each member for a given number of epochs

    Args:
        n_models: population size
        batch: the first minibatch of the training set
        initial_model: initial model to mutate
        init_weights_path: path to save model parameters
        sch_epochs: successive halving epochs
        descriptors: a list to keep model descriptor for each member of the population
        trainloader: train set
        n_epochs_each: initial short training epochs for each member of the population
        performance: a list to keep track of performance of each network in the population
        validloader: validation set
        folder_out: folder number to save the model
        savepath: path to save the model

    Returns:
        Returns a list of performances of the models and model descriptors
    """
    for model_idx in range(0, n_models):

        time_overall_s = time.time()

        # Load the parameters of the initial model of this round. AKA parent network.
        pytorch_model = ConvNet(initial_model['model_descriptor'])
        pytorch_model.cuda()
        pytorch_model.load_state_dict(torch.load(init_weights_path), strict=False)

        # Create the meta dictionary for the model
        model = {'pytorch_model': pytorch_model,
                'model_descriptor': copy.deepcopy(initial_model['model_descriptor']),
                'topo_ordering': pytorch_model.topo_ordering}

        # Set the lr scheduler
        model['pytorch_model'].scheduler = optim.lr_scheduler.CosineAnnealingLR(model['pytorch_model'].optimizer, T_max=sch_epochs, eta_min=0.0, last_epoch=-1)
    
        # Add the new model's description dictionary
        descriptors.append(model['model_descriptor'])

        # Keep track of time spent: [overall, mutations, training]
        times = [0, 0, 0]
        
        # Given the parent network, return a mutated new network
        times, model = mutate_network(n_mutations, model, model_idx, batch, max_n_params, times)
        

        time_train_s = time.time()
        
        # Initial short training of the child, first step of the successive halving
        model['pytorch_model'].fit(trainloader, train_type = 'child' ,epochs=n_epochs_each)

        time_train_e = time.time()
        times[2] = times[2] + (time_train_e - time_train_s)

        # Evaluate the child network, keep track of the performance
        performance[model_idx] = model['pytorch_model'].evaluate(validloader)

        # Save the model and other related information about it
        pytorch_total_params_child = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)
        with open(folder_out + "performance.txt", "a+") as f_out:
            f_out.write('child ' + str(model_idx) + ' performance ' +str(performance[model_idx]) + ' num params ' + str(pytorch_total_params_child) +'\n')

        torch.save(model['pytorch_model'].state_dict(), savepath + 'model_' + str(model_idx))
        torch.save(model['pytorch_model'].scheduler.state_dict(), savepath + 'model_' + str(model_idx)+'_scheduler')
        torch.save(model['pytorch_model'].optimizer.state_dict(), savepath + 'model_' + str(model_idx)+'_optimizer')

        descriptors[model_idx] = copy.deepcopy(model['model_descriptor'])

        time_overall_e = time.time()

        times[0] = times[0] + (time_overall_e - time_overall_s)

        np.savetxt(savepath + 'model_' + str(model_idx) + '_times', times)

        descriptor_file = open(savepath + 'model_' + str(model_idx) + '_model_descriptor.txt', 'w')

        for layer in model['model_descriptor']['layers']:
            layer_str = str(layer)
            descriptor_file.write(layer_str + "\n")
        descriptor_file.close()
        del model['pytorch_model']
        del model
        torch.cuda.empty_cache()
    return performance, descriptors

def successive_haling(n_children, n_epochs_train_children, savepath ,sorted_children, descriptors, trainloader, validloader, folder_out, performance):
    """
    Method which performes successive halving given a population of child networks 
    
    Args:
        n_children: number of child models
        n_epochs_train_children: number of epochs to train each child, this value updates during the search
        savepath: path to save the models for further training or analysis
        sorted_children: sorted network ids, with respect to their performances
        descriptors: model descriptors
        trainloader: training set
        validloader: validation set
        folder_out: folder number to save the results
        performance: performances of the population elements
    Returns:
        Returns the winner model
    """
    # Continue searching until we find the winner model    
    while n_children > 1:

        best_children = sorted_children[(n_children // 2):]
        
        # Increase the training budget for the next SH rounds
        n_epochs_train_children = n_epochs_train_children * 2
        

        for child in best_children:
            
            # Load the child model parameters
            pytorch_model = ConvNet(descriptors[child])
            pytorch_model.cuda()
            pytorch_model.load_state_dict(torch.load(savepath + 'model_' + str(child)), strict=False)
            pytorch_model.scheduler.load_state_dict(torch.load(savepath + 'model_' + str(child)+'_scheduler'))
            pytorch_model.optimizer.load_state_dict(torch.load(savepath + 'model_' + str(child)+'_optimizer'))
            model = {'pytorch_model': pytorch_model,
                    'model_descriptor': copy.deepcopy(descriptors[child]),
                    'topo_ordering': pytorch_model.topo_ordering}

            # Train a child
            model['pytorch_model'].fit(trainloader, train_type = 'child' ,epochs=n_epochs_train_children)

            # Evaluate a child
            performance[child] = model['pytorch_model'].evaluate(validloader)

            # Save
            pytorch_total_params_child = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)
            with open(folder_out + "performance.txt", "a+") as f_out:
                f_out.write('child ' + str(child) + ' performance ' +str(performance[child]) + ' num params ' + str(pytorch_total_params_child) + '\n')

            torch.save(model['pytorch_model'].state_dict(), savepath + 'model_' + str(child))
            torch.save(model['pytorch_model'].scheduler.state_dict(), savepath + 'model_' + str(child)+'_scheduler')
            torch.save(model['pytorch_model'].optimizer.state_dict(), savepath + 'model_' + str(child)+'_optimizer')

            del model['pytorch_model']
            del model
            torch.cuda.empty_cache()

        # Sort the population with respect to the performance
        temp_children_array = np.argsort(performance)
        sorted_children = []

        # Get only the best halves' members for the next iteration
        for t in temp_children_array:
            if t in best_children:
                sorted_children.append(t)

        n_children = len(sorted_children)


    # Load the winner child, save its performance and return it
    the_best_child = sorted_children[0]

    pytorch_model = ConvNet(descriptors[the_best_child])
    pytorch_model.cuda()
    pytorch_model.load_state_dict(torch.load(savepath + 'model_' + str(the_best_child)), strict=False)
    model = {'pytorch_model': pytorch_model,
            'model_descriptor': copy.deepcopy(descriptors[the_best_child]),
            'topo_ordering': pytorch_model.topo_ordering}
    
    with open(folder_out + "performance.txt", "a+") as f_out:
        f_out.write("****************************\n") 

    return model, performance[sorted_children[0]]
        
def get_winner_child_network(n_models, batch, n_mutations, trainloader, validloader, n_epochs_total, initial_model, savepath, folder_out):
    """
    Method to generate, train and select the best model
    
    Args:
        n_models: number of child models
        batch: the first batch of the training set
        n_mutations: number of mutations/network operators to be applied per model_descriptor
        trainloader: training set
        validloader: validation set
        n_epochs_total: number of epochs for training in total
        initial_model: current best model_descriptor
        savepath: save the results 
        folder_out: save other analysis files
    Returns:
        Returns the winner model
    """

    # Prepare the saving path for inital weights
    init_weights_path = savepath + 'ini_weights'

    # Save the initial model
    torch.save(initial_model['pytorch_model'].state_dict(), init_weights_path)

    # Keep track of performance and model descriptors
    performance = np.zeros(shape=(n_models,))
    descriptors = []

    # Get the total successive halcing steps required for the scheduler
    n_minibatches = len(trainloader)
    sch_epochs = get_sh_epochs(n_epochs_total,n_models,n_minibatches)

    # Given the initial network, generate a population of child networks and perform their initial training.    
    # n_models, batch, initial_model, init_weights_path, sch_epochs, descriptors, trainloader, n_epochs_each, performance, validloader, folder_out, savepath
    performance, descriptors = generate_population_and_train(n_models, batch, initial_model, init_weights_path, sch_epochs, descriptors, trainloader, n_epochs_total, performance, validloader, folder_out, savepath)
    
    # Sort the networks with respect to their performance
    sorted_children = np.argsort(performance)
    n_children = len(sorted_children)
    
    # Seach for the winner of the experiment using successive halving
    winner, winner_performance = successive_haling(n_children, n_epochs_total, savepath ,sorted_children, descriptors, trainloader, validloader, folder_out, performance)
    return winner

def final_training(model, folder_out):
    """
    Function to train the winner model 

    Args:
        model: the winner model of the experiment
        folder_out: folder to save the model
    Returns:
    """
    print('final training')

    # Load training data without validation part before final training
    trainloader_final, _, testloader_final = utils.prepare_data(valid_frac=0.0)
    
    # Change lr for the final training 
    model['pytorch_model'].optimizer.param_groups[0]['initial_lr'] = lr_final
    model['pytorch_model'].optimizer.param_groups[0]['lr'] = lr_final

    # Train the winner model
    model['pytorch_model'].fit(trainloader_final, train_type = 'winner', epochs=epoch_final)

    # Evaluate the performance
    performance = model['pytorch_model'].evaluate(testloader_final)
    final_num_params = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)

    # Save
    with open(folder_out + "performance.txt", "a+") as f_out:
        f_out.write('final perf ' + str(performance) + ' final number of params ' + str(final_num_params))

    torch.save(model['pytorch_model'].state_dict(), folder_out + 'best_model')
    descriptor_file = open(folder_out + 'best_model_descriptor.txt', 'w')
    for layer in model['model_descriptor']['layers']:
        layer_str = str(layer)
        descriptor_file.write(layer_str + "\n")
    descriptor_file.close()
