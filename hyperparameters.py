import os
import shutil
import numpy as np

mutation_time_limit_hours = 23
n_models = 8  # Number of child models per generation
n_mutations = 5  # Number of mutations applied per generation
n_epochs_total = 5  # Training budget for the first generation
epoch_final = 200  # Epochs for final training, if we have time left => 23 hours limit
lr_final = 0.025
n_experiments = 8
max_n_params = 20*10**6 # Similar to mutation time limit
expfolder = "./results_sh/"
shutil.rmtree('./results_sh', ignore_errors=True)
os.makedirs(expfolder)
mutations_probs = np.array([1, 1, 1, 1, 1, 0])