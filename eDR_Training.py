# -*- coding: utf-8 -*-
"""
Created on Mar 23 13:27:04 2022

@author: mobecks

Training of enhanced deep regression (eDR) neural network on real world dataset.
Including ray tune to automatically find best architecture and best HP.

Additions: 
    - Z2 shim included
    - preparation to load randomly acquired dataset
        - include past "actions"
    - shim weighting
    - Replaced CNN with ConvLSTM
    - combine datasets
    - variable sequence lengths during training
"""

import os
import time
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset
import ray
from ray import tune

# install dev version: python -m pip install git+https://github.com/jjhelmus/nmrglue.git@6ca36de7af1a2cf109f40bf5afe9c1ce73c9dcdc

import argparse
parser = argparse.ArgumentParser(description="Run")
parser.add_argument("--raytuning",type=int,default=0) 	# 0 for standard training
input_args = parser.parse_args()

import sys
MYPATH = '' 			#TODO: insert path to scripts
DATASET_PATH_1 = ''		#TODO: insert path to dataset (reference sample)
DATASET_PATH_3 = ''		#TODO: insert path to dataset (CuSO4)

sys.path.append(MYPATH+'Utils/')
import utils
import utils_IO
from models import ConvLSTM

DATAPATH = MYPATH+'Learning/data/'
if not os.path.exists(DATAPATH):
    os.mkdir(DATAPATH)

initial_config = {
        'id': 19,                    	# id of trained model to trace results
        "set":'random',             	# high level descriptor. random data used.
        'model': 'ConvLSTM',        	# ConvLSTM
        "nr_shims": 4,
        "input_channels": 10,       	# sequence length for LSTM
        'combine_datasets': False,   	# combine datasets into one dataloader
        'train_variable_length': True,  # fixed 'input_channels' length or variable from (4 to 10)
        # Augmentation
        "shift_augm": 4,           		# z0 shift (shift xaxis)
        "shift_type": 'normal',    		# 'complex' shifts each spectrum in the sequence by different value
        "label_noise": .1,           	# uniform( 1/range * label_noise )
        'label_noise_type': 'normal',   # differnt noise to each spectrum in sequence = complex, same noise = normal
        'interaction_noise': .1,      	# label noise in matrix representation to simulate shim interactions. uniform( 1/range * label_noise )
        "phase0_range": .5,         	# 220321-164037 -> 10. percentile -0.36 | 90. percentile 0.85
        "phase1_range": 0,          	# 211215-171500 -> 0
        'awgn_SNR': 30,             	# SNR of additive white gaussian noise
        # Data scaling
        "downsample_factor": 1,     	# dont downsample, but select ROI instead!
        'ROI_min': [16000],  			# calculated via mean_idx(signal(2048p)>2*noise) * downsamplefactor. ROI_min for [h2o]
        "max_data": 1e5,            	# first scaling of intensity. Second scaling is if "scale_sample"
        'scale_sample': True,       	# scale first sample in sequence to one (instead of global max)
        'shim_weightings': [1.2,1,2,18,0,0,0,0,0,0,0,0,0,0,0],  # Shim weighting. range * weighting = shim values
        'acq_range': 50,            	# range of absolute shim offsets for highest-impact shim (with weighting=1)
        # Architecture
        "num_layers_cnn": 5,			# CNN layers
        "kernel_size": 19,				# CNN kernel size
        "stride": 2, 					# CNN stride
        "pool_size": 1,					# CNN pooling size
        "filters_conv": 64,				# CNN number of kernels
        "drop_p_conv":0.2,				# CNN dropout rate
        "drop_p_fc":0.2,				# FC dropout rate
        'hidden_size': 1024,			# FC & LSTM hidden size
        "num_layers_lstm": 3,			# LSTM layers
        # HP
        "LR": 5e-4,
        'loss': 'Huber',
        "batch_size": 32,
        "epochs": 100,              	# if 'train_variable_length' epochs = total nr of epochs
        'optimizer': 'adam',
        "enable_subbatches": False, 	# allows training on small GPUs
        'clip_grad': False,         	# gradient clipping 
        'max_norm': 2,              	# clipping max norm
    }

#if GPU runs out of memory consider changing the batch size
#https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
#if BATCH_SIZE is to big for GPU, enable sub-batches by setting TRUE
if initial_config["enable_subbatches"] == True:
    divisor = 8 #has to be divider of inital batch size
    BATCH_SIZE = int(initial_config["batch_size"] / divisor)

# reproduceability
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 5
seed_everything(seed)

#%% Pre-load data and store in pickle files. (Big time saving for training!)

skip_pickling_arr = [False, False,]
ROI_idx_arr = [0,0]
ds_paths = [DATASET_PATH_1,DATASET_PATH_3]
pickle_names = ['set1', 'set2',]                # TODO change according to your will
ROLLS = [0,0]

for idx in range(len(ROLLS)):

    skip_pickling = skip_pickling_arr[idx]
    if skip_pickling != True:
        data_all, labels_all, xaxis = utils_IO.get_dataset(
						ds_paths[idx], target_def = 'relative', normalize=False, downsamplefactor=1,
						ROI_min=initial_config['ROI_min'][ROI_idx_arr[idx]], nr_shims=initial_config['nr_shims'])
    
        data = data_all/initial_config['max_data']
        
        # normalize to [-1,1]
        labels = labels_all/\
            (np.multiply(initial_config['acq_range'], initial_config['shim_weightings'][:initial_config['nr_shims']]))
        
        # store as pickle. 
        with open (os.path.dirname(os.path.abspath(DATASET_PATH_1))+'/preloaded_pickle/{}.pickle'.format(pickle_names[idx]), "wb") as f:
            pickle.dump([data, labels, xaxis],f)

#%% Load data from pickle

def load_data(PATHNAME):
    # load whole package from disc. Save iteration time.
    with open(os.path.dirname(os.path.abspath(DATASET_PATH_1))+PATHNAME,'rb') as f:
        [data, labels, dic] = pickle.load(f)          
    data_all_id = ray.put(data)
    labels_all_id = ray.put(labels)
    return data_all_id, labels_all_id

path_h2o = '/preloaded_pickle/set1.pickle'
data_h2o_id, labels_h2o_id = load_data(path_h2o)

path_cuso4 = '/preloaded_pickle/set2.pickle'
data_cuso4_id, labels_cuso4_id = load_data(path_cuso4)


#%% train/val/test splitting (80/10/10 %)

def create_sets(data_all_id, labels_all_id, seed=52293):

    data_all = ray.get(data_all_id)
    labels_all = ray.get(labels_all_id)
    
    # !!! prevent leakage
    np.random.seed(seed)

    trainsize, valsize = int(len(data_all)*0.8), int(len(data_all)*0.1)
    set_sizes = [trainsize, valsize, int(len(data_all)-trainsize-valsize)]
    idxs_1dist = np.arange(0, len(data_all), 1)

    #randomly assign to train set
    rand_idxs = np.random.choice( np.arange(0, len(data_all)), size = set_sizes[0], replace=False)
    data = np.array([ data_all[i] for i in rand_idxs])
    labels = np.array([ labels_all[i] for i in rand_idxs])
    data_id = ray.put(data)         # put into ray data handler. Torch dataset does not allow large arguments
    labels_id = ray.put(labels)

    # create val set of remaining
    remaining = list(set(np.arange(0,len(data_all)))  - set(rand_idxs))
    rand_idxs_val = np.random.choice( remaining , size = set_sizes[1], replace=False)
    data_val = np.array([ data_all[i] for i in rand_idxs_val])
    labels_val = np.array([ labels_all[i] for i in rand_idxs_val])
    data_val_id = ray.put(data_val)
    labels_val_id = ray.put(labels_val)

    #create test set of remaining
    remaining = list(set(remaining)  - set(rand_idxs_val))
    rand_idxs_test = np.random.choice( remaining , size = set_sizes[2], replace=False)
    data_test = np.array([ data_all[i] for i in rand_idxs_test])
    labels_test = np.array([ labels_all[i] for i in rand_idxs_test])
    data_test_id = ray.put(data_test)
    labels_test_id = ray.put(labels_test)

    return data_id, labels_id, data_val_id, labels_val_id, data_test_id, labels_test_id


#%%
class MyDataset(Dataset):
    def __init__(self, data_id, labels_id, config, transform=False):
        self.data = ray.get(data_id)  
        self.labels = ray.get(labels_id)
        self.transform = transform
        self.config = config
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        noise_step = np.ones(self.config["nr_shims"])/self.config['acq_range'] 	# define noise step as 1 discrete increment of shims
        arr = np.arange(self.data.shape[0])     								# allocate array for indices
        idxs_rand = np.random.choice(np.delete(arr,idx), replace=False, size=(self.config['input_channels']-1)) # get random indices of data to load for sequence
        idxs = np.append(idx, idxs_rand) 										# append to idx of __getitem__

        # Change labels and concat with data for DQN_Fuse
        labels_rand = self.labels[idxs_rand] - self.labels[idx] 							# relative to unshimmed
        labels_out = np.concatenate((np.zeros([1,self.config['nr_shims']]), labels_rand)) 	# Optional: change architecture? 
        
        if self.config['scale_sample']: sc_factor = self.data[idx].max() 					# scale initial unshimmed to [~0,1]. Rest relative
        else: sc_factor = 1
                
        if self.transform:             
            # homogen shift
            if self.config["shift_type"] == 'normal':
                batch = torch.roll(torch.tensor(self.data[idxs]),
                        np.random.randint(-int(self.config["shift_augm"]),int(self.config["shift_augm"]))).float()
            #heterogen shift
            if self.config["shift_type"] == 'complex':
                batch = torch.tensor(np.zeros([ self.config["input_channels"] , 2048]))
                shifts =np.random.randint(-int(self.config["shift_augm"]), int(self.config["shift_augm"]), size=self.config["input_channels"])
                for ix, s in enumerate(shifts):
                    batch[ix] = torch.roll(torch.tensor(self.data[idxs[ix]]), s).float()
            # add noise to the labels (here to relative shim offsets. target label is modified at return)
            if self.config["label_noise"]!=0:
                # target noise
                noise = noise_step*np.random.uniform(-(self.config["label_noise"]),(self.config["label_noise"]),
                                          size=self.config["nr_shims"])
                # offset noise (step not uniform)
                if self.config['label_noise_type']=='complex':
                    labels_out=labels_out+self.config["label_noise"]*np.concatenate((np.zeros([1,self.config['nr_shims']]),
                                np.random.uniform(-noise_step,noise_step,size=(self.config['input_channels']-1,self.config['nr_shims']))))
                elif self.config['label_noise_type']=='normal':
                    labels_out=labels_out+self.config["label_noise"]*np.concatenate((np.zeros([1,self.config['nr_shims']]),
                                np.tile( np.random.uniform(-noise_step,noise_step,size=(1,self.config['nr_shims'])),
                                        self.config['input_channels']-1).reshape(self.config['input_channels']-1,-1)))
            else:
                noise = 0  
            if self.config['interaction_noise']!=0:
                # matrix for interactions between shims
                range_interactions = self.config["label_noise"]*noise_step
                self.interactions = np.random.uniform(-range_interactions,range_interactions,
                                                      size = (self.config["nr_shims"], self.config["nr_shims"]))
                self.interactions = (self.interactions + self.interactions.T)/2     # make symmetric
                np.fill_diagonal(self.interactions, 1)  
            else:
                self.interactions = np.zeros((self.config["nr_shims"], self.config["nr_shims"]))
                np.fill_diagonal(self.interactions,1) # no interactions
            if self.config['awgn_SNR']!=None:
                std = utils.get_noise_avg(batch[0].numpy(),self.config['awgn_SNR'])
                awgn = np.random.normal(0,std, size=batch.shape)
            else:
                awgn = 0
            # linear phase distortion. Adapted from nmrglue to handle torch object.
            if self.config["phase0_range"] != 0:
                p0 = np.random.uniform(-self.config["phase0_range"],self.config["phase0_range"]) * np.pi / 180.  # convert to radians
                p1 = np.random.uniform(-self.config["phase1_range"],self.config["phase1_range"]) * np.pi / 180.
                size = batch.shape[-1]
                apod = torch.exp(1.0j * (p0 + (p1 * torch.arange(size) / size)))
                batch = (apod*batch).real
                    
            return  torch.cat(((batch+awgn)/sc_factor,
                    torch.tensor(np.matmul(labels_out,self.interactions))), dim=-1),\
                    torch.tensor(np.matmul(self.interactions,self.labels[idx])+noise).float()

        return torch.cat((torch.tensor(self.data[idxs]/sc_factor),torch.tensor(labels_out)),dim=-1), torch.tensor(self.labels[idx]).float()


#%% training

def train(config, checkpoint_dir=None, raytune=False):
    sys.path.append(MYPATH+'Utils/')
    from models import ConvLSTM, weights_init

    if config['model'] == 'ConvLSTM':
        model = ConvLSTM(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                     output_size=config["nr_shims"], num_layers_lstm=2,
                 num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                 stride=config["stride"], kernel_size=config["kernel_size"])

    #model.apply(weights_init)

    device = "cuda" if torch.cuda.is_available() else "cpu"
# =============================================================================
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
# =============================================================================
    model.to(device)
    
    if config['loss']=='MSE': criterion = nn.MSELoss()
    if config['loss']=='MAE': criterion = nn.L1Loss()
    if config['loss']=='Huber': criterion = nn.HuberLoss()
    
    if config["optimizer"]=='SGD':optimizer = torch.optim.SGD(model.parameters(), lr=config["LR"], momentum=.9)
    elif config["optimizer"]=='adam':optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    elif config["optimizer"]=='adagrad':optimizer = torch.optim.Adagrad(model.parameters(), lr=config["LR"])

    error_train = []
    error_val = []
    acc_train = []
    acc_val = []

    if config['train_variable_length']:  # if true, loop training 
        start = 4
        epoch_counter = 0
    else: 
        start = config['input_channels'] # force to one training run only   
        start_epoch = 0 
       
    # loop training. If 'train_variable_length'==False -> only 1 run
    for i in range(start, config['input_channels']+1, 2): # every 2nd step
    #for i in range(config['input_channels'], start-1, -2): # every 2nd step AND go backwards
        local_config = config.copy() # = config but 'input_channel' change for dataset
    
        # adapt length and reduce epochs
        if config['train_variable_length']: 
            scaler = 4
            local_config['input_channels'] = i # overwrite
            end_epoch = int(epoch_counter * config['epochs']/scaler + config['epochs']/scaler)
            start_epoch = int(epoch_counter * config['epochs']/scaler) #sliding window for epoch for ray saving
            epoch_counter+=1

        datasets = []
        datasets_val = []
    
        # load different sets of data. "Base" dataset = Reference data
        data_id, labels_id, data_val_id, labels_val_id, data_test_id, labels_test_id = create_sets(data_h2o_id, labels_h2o_id)
        dataset_h2o = MyDataset(data_id, labels_id, local_config, True)
        dataset_val_h2o = MyDataset(data_val_id, labels_val_id, local_config)
        datasets.append(dataset_h2o)
        datasets_val.append(dataset_val_h2o)
    
        if config['combine_datasets']:
            data_id, labels_id, data_val_id, labels_val_id, data_test_id, labels_test_id = create_sets(data_cuso4_id, labels_cuso4_id)
            dataset_cuso4 = MyDataset(data_id, labels_id, local_config, True)
            dataset_val_cuso4 = MyDataset(data_val_id, labels_val_id, local_config)   
            datasets.append(dataset_cuso4)
            datasets_val.append(dataset_val_cuso4)               
       
        concatenated_dataset_train = torch.utils.data.ConcatDataset(datasets)
        concatenated_dataset_val = torch.utils.data.ConcatDataset(datasets_val)
    
        train_loader = torch.utils.data.DataLoader(concatenated_dataset_train, batch_size=int(config["batch_size"]), shuffle=True)
        val_loader = torch.utils.data.DataLoader(concatenated_dataset_val, batch_size=int(config["batch_size"]), shuffle=True)
    
        # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
        # should be restored.
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    
        for epoch in range(start_epoch, end_epoch):
            if not raytune: print("Epoch #", epoch)
            start_t = time.time()
            for mode, dataloader in [("train", train_loader), ("val", val_loader)]:
    
                if mode == "train":
                    model.train()
                else:
                    model.eval()
                    state = model.state_dict() # track state if manual reset is wanted
                    state_error = error_val[-1] if error_val else 0
    
                runningLoss = 0
                total = 0
    
                for i_batch, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.float(), targets.float()
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    if config['model']=='ConvLSTM':
                        (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
                        h0, c0 = h0.to(device), c0.to(device)     
                        outputs, hidden = model(inputs, (h0,c0))                  
                        outputs = outputs[:,-1]
                    else: # normal CNN forward behaviour
                        outputs = model(inputs)
                        
                    loss = criterion(outputs, targets)
                    
                    runningLoss += loss.item() * inputs.shape[0]
                    total += inputs.shape[0]
    
                    if mode == "train":
                        loss.backward()
                        if config['clip_grad']:
                            nn.utils.clip_grad_norm_(model.parameters(), config['max_norm'])
                        if initial_config["enable_subbatches"] == True:
                            if (i_batch+1)%divisor == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                        else:
                            optimizer.step()
                            optimizer.zero_grad()
    
                (error_train if mode == "train" else error_val).append(runningLoss / total)
    
            if raytune:
                # Here we save a checkpoint. It is automatically registered with
                # Ray Tune and will potentially be passed as the `checkpoint_dir`
                # parameter in future iterations.
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
                tune.report(loss = error_val[-1], loss_train = error_train[-1]  )
    
    
            end_t = time.time()
            if not raytune:
                print('Train error: ', round(error_train[-1],4))
                print('Val error: ', round(error_val[-1],4))
                print('Time epoch: ', round(end_t - start_t))

    return model, error_train, error_val
	
# %% Testing

def test_best_model(best_trial=None, raytune=False, model=None, config=None):
    sys.path.append(MYPATH+'Utils/')
    from models import ConvLSTM, weights_init

    # allow local and raytune runs.
    if raytune:
        config = best_trial.config
    else:
        config = config

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if raytune:
        if config['model'] == 'ConvLSTM':
            model = ConvLSTM(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                         output_size=config["nr_shims"], num_layers_lstm=2,
                     num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                     stride=config["stride"], kernel_size=config["kernel_size"])
        checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    datasets_test = []

    data_id, labels_id, data_val_id, labels_val_id, data_test_id, labels_test_id = create_sets(data_h2o_id, labels_h2o_id)
    dataset_test_h2o = MyDataset(data_test_id, labels_test_id, config)
    datasets_test.append(dataset_test_h2o)
 
    if config['combine_datasets']:
        data_id, labels_id, data_val_id, labels_val_id, data_test_id, labels_test_id = create_sets(data_cuso4_id, labels_cuso4_id)
        dataset_test_cuso4 = MyDataset(data_test_id, labels_test_id, config)
        datasets_test.append(dataset_test_cuso4)
    
    concatenated_dataset_test = torch.utils.data.ConcatDataset(datasets_test)
        
    test_loader = torch.utils.data.DataLoader(concatenated_dataset_test, batch_size=int(config["batch_size"]), shuffle=True)

    if config['loss']=='MSE': criterion = nn.MSELoss()
    if config['loss']=='MAE': criterion = nn.L1Loss()
    if config['loss']=='Huber': criterion = nn.HuberLoss()

    error_test = 0
    total = 0
    mae = np.array([])
    
    # plot predictions vs targets. See figure in spplementary of paper #1
    plt.figure(figsize=(6,4))
    counter = 0
    samples = 10

    for i_batch, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)

        if config['model']=='ConvLSTM':
            (h0,c0) = model.initHidden(batch_size=inputs.shape[0]) # TODO CHECK
            h0, c0 = h0.to(device), c0.to(device)     
            outputs, hidden = model(inputs, (h0,c0))                  
            outputs = outputs[:,-1]
        else:
            outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)

        error_test += loss.item() * inputs.shape[0]
        total += inputs.shape[0]

        mae = np.append(mae, abs(outputs.cpu().detach().numpy())-abs(targets.cpu().detach().numpy()))
        
        if counter <= samples: 
            plt.plot(counter, outputs.cpu().detach().numpy()[0,0]*config['shim_weightings'][0]*50, 'b.',)
            plt.plot(counter, targets.cpu().detach().numpy()[0,0]*config['shim_weightings'][0]*50, 'g.',)
            plt.plot(counter, outputs.cpu().detach().numpy()[0,1]*config['shim_weightings'][1]*50, 'bx',)
            plt.plot(counter, targets.cpu().detach().numpy()[0,1]*config['shim_weightings'][1]*50, 'gx',)
            plt.plot(counter, outputs.cpu().detach().numpy()[0,2]*config['shim_weightings'][2]*50, 'bs',)
            plt.plot(counter, targets.cpu().detach().numpy()[0,2]*config['shim_weightings'][2]*50, 'gs',)
            plt.plot(counter, outputs.cpu().detach().numpy()[0,3]*config['shim_weightings'][3]*50, 'bv')
            plt.plot(counter, targets.cpu().detach().numpy()[0,3]*config['shim_weightings'][3]*50, 'gv')
            plt.annotate(int(np.mean(abs(abs(outputs.cpu().detach().numpy()*32768/100)[0]
                        -abs(targets.cpu().detach().numpy()*32768/100)[0]))), [counter,10100], ha='center')
            counter+=1
       
    # generate legend with last result
    plt.plot(counter-1, outputs.cpu().detach().numpy()[0,0]*config['shim_weightings'][0]*50, 'b.', label='$\hat{y}_X$')
    plt.plot(counter-1, targets.cpu().detach().numpy()[0,0]*config['shim_weightings'][0]*50, 'g.', label='$y_X$')
    plt.plot(counter-1, outputs.cpu().detach().numpy()[0,1]*config['shim_weightings'][1]*50, 'bx', label='$\hat{y}_Y$')
    plt.plot(counter-1, targets.cpu().detach().numpy()[0,1]*config['shim_weightings'][1]*50, 'gx', label='$y_Y$')
    plt.plot(counter-1, outputs.cpu().detach().numpy()[0,2]*config['shim_weightings'][2]*50, 'bs', label='$\hat{y}_Z$')
    plt.plot(counter-1, targets.cpu().detach().numpy()[0,2]*config['shim_weightings'][2]*50, 'gs', label='$y_Z$')
    plt.plot(counter-1, outputs.cpu().detach().numpy()[0,3]*config['shim_weightings'][3]*50, 'bv', label='$\hat{y}_{Z2}$')
    plt.plot(counter-1, targets.cpu().detach().numpy()[0,3]*config['shim_weightings'][3]*50, 'gv', label='$y_{Z2}$')
    
    plt.legend(frameon=True)
    plt.ylim([-500,500])
    plt.xlim([-1,samples+1])
    plt.ylabel("Shim value")
    plt.xlabel("Test sample")
    plt.annotate('$=MAE$', [counter,1030], ha='center')
    plt.xticks(np.arange(samples+1), np.arange(samples+1))
    #plt.savefig(DATAPATH+'/fig_MAE_explanation.pdf')

    mae = np.reshape(mae, (-1,config["nr_shims"]) )

    print('\n NN: MAE test set {} +/- {}'.format(np.mean(np.abs(mae)), np.std(np.abs(mae))))

    mae_ax = np.mean(np.abs(mae), axis=0)
    print('\n NN: MAE test set (axis-specific): {}'.format(mae_ax))

    if not raytune:
        plt.figure()
        plt.plot(error_train, label = "Train Error")
        plt.plot(error_val, label = "Val Error")
        plt.plot(initial_config["epochs"]-1, error_test/total, 'x', label = "'Test' Error")
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training loss (MSE) for regression NN')
        plt.savefig(DATAPATH + '/loss_DR_Z2_{}_id{}.png'.format(initial_config["set"],initial_config["id"]))
        plt.show()

    torch.save(model.state_dict(), DATAPATH + '/model_DR_Z2_{}_id{}.pt'.format(initial_config["set"],initial_config["id"]))
    import json
    with open(DATAPATH + '/config_DR_Z2_{}_id{}.json'.format(initial_config["set"],initial_config["id"]), 'w') as f:
        json.dump(config,f)

if input_args.raytuning == 0:
    model, error_train, error_val = train(config=initial_config)
    test_best_model(model=model,config=initial_config)

#state = torch.load(DATAPATH + '/model_{}.pt'.format('multiregression'))
#model.load_state_dict(state)


#%% HyperOpt with ray tune
from ray.tune.schedulers import ASHAScheduler   # early stopping
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator

#https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def main(num_samples=10, max_num_epochs=150, gpus_per_trial=.5, num_workers=8):
    # search space
    # TODO change according interest
    config = {
        'model': tune.choice(['ConvLSTM']),
        'input_channels': tune.choice([4,6,8,10,12]),
        "num_layers_cnn": tune.choice([3,4,5]),
        "num_layers_lstm": tune.choice([1,2,3]),
        "kernel_size": tune.choice([11,19,21,31,41,51,71]),
        #"stride": tune.choice([1,2]),
        "pool_size": tune.choice([1,2]),           
        #"LR": tune.loguniform(1e-5, 1e-3, 5e-4),                                 
        #"batch_size": 32,                           
        #'optimizer': tune.choice(['adam', 'SGD']),
        #"shift_augm": tune.choice([16,32,64]),                         
        #"drop_p_conv": tune.choice([.1,.2,.5]),
        #"drop_p_fc":  tune.choice([.1,.2,.5]),
        #'filters_conv': tune.choice([32,64]),
        #'hidden_size': tune.choice([256,1024]),
        #'label_noise':  tune.choice([.1,.5,1]),
    }
    
    config = merge_two_dicts(initial_config, config)   #merge configs

    
    # early stoppping scheduler
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=5,         # min. nr of trained epochs
        reduction_factor=2)
                       
    # random search
    searcher =BasicVariantGenerator()

    result = tune.run(
        tune.with_parameters(train, raytune=True),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        name="raytune_{}".format(initial_config['set']),
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir = '/'.join(os.path.normpath(DATAPATH).split(os.sep)[:-5]) + '/ray_results/',
        search_alg=searcher,
        raise_on_failed_trial  = False, # allow errors
        keep_checkpoints_num = 1, # reduce disc load
        checkpoint_score_attr = 'min-loss',
    )

    best_trial = result.get_best_trial("loss", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.tune.utils.util import force_on_current_node
        remote_fn = force_on_current_node(ray.remote(tune.with_parameters(test_best_model, raytune=True)))
        ray.get(remote_fn.remote(best_trial))
    else:
        test_best_model(best_trial, raytune=True)

    return result

if input_args.raytuning == 1:
    result = main(num_samples=150,max_num_epochs=initial_config["epochs"], gpus_per_trial=.2)
    df = result.dataframe()

    df.to_excel(DATAPATH + '/raytune_results_{}_{}.xlsx'.format(initial_config["set"],datetime.today().strftime('%Y-%m-%d')))
    
# log results via 
# $tensorboard --logdir ~/ray_results