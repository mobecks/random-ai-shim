# -*- coding: utf-8 -*-
"""
Created on Tue 29 Mar 08:26:01 2022

@author: mobecks

live Deep Regression with random shim values and ConvLSTM
on Magritek 80MHz benchtop

run over 100 random distortions
"""

import os
import sys
import json
import glob
import pickle
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng
from datetime import datetime
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from scipy import signal, optimize
from numpy.polynomial.polynomial import polyfit


MYPATH = ''             # TODO: insert path to scripts
DATAPATH = ''           # TODO: insert path for data
sys.path.append(MYPATH)
import utils_Spinsolve

import argparse
parser = argparse.ArgumentParser(description="Run enhanced deep regression")
parser.add_argument("--verbose",type=int,default=2) # 0 for no output, 1 for minimal output, 2 for max output
input_args = parser.parse_args()

#plt.style.use(['science', 'nature', 'high-contrast'])
##plt.rcParams.update({"font.family": "sans-serif",})

import warnings
if input_args.verbose == 2: warnings.filterwarnings("ignore")

#https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

initial_config = {
        'id': 19,                   # model's id
        "sample": 'Ref',            # Ref (=5%), Ref10 (10%), H2ONic400 (1ml D2O with 400mg Nicotinamid)
        'postshim': True,           # shim after experiment to guarantee same starting points
        'full_memory': True,        # store whole spectrum in additional memory
        "set":'random',
        "downsample_factor": 1,
        'ROI_min': 16000,           
        "max_data": 1e5,            # pre-scaling
        'scale_sample': True,       # scale first sample in sequence to one (instead of global max)
        'shim_weightings': [1.2,1,2,18,0,0,0,0,0,0,0,0,0,0,0],  # Shim weighting. range * weighting = shim values
        'acq_range': 50,            # range of absolute shim offsets for highest-impact shim (weighting=1)
        "drop_p_fc": 0.0,
        "drop_p_conv": 0.0,
    }     

with open(DATAPATH + '/eDR/models/config_DR_Z2_{}_id{}.json'.format(initial_config['set'],initial_config["id"])) as f:
    config = json.load(f)

config = merge_two_dicts(initial_config, config)   #merge configs


# # overwrite ROI for different peaks
if 'H2O' in config['sample'] or 'Ref' in config['sample']:    
    config['ROI_min'] = 16000
    ROLL = 0
    PLT_X = [80,120]
elif 'TSP' in config['sample']:    
    config['ROI_min'] = 18500
    ROLL = 0
    PLT_X = [450,550]

sampling_points = 32768
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Generate list of random distortions for evaluation
seed = 45612
nr_evaluations = 100
np.random.seed(seed)
gauss_noise = np.random.normal(0, 1/3,size=(nr_evaluations,config['nr_shims']))
random_distortions = (config['acq_range']*gauss_noise*config['shim_weightings'][:config['nr_shims']]).astype(int) #discrete uniform

class Arguments():
    def __init__(self):
        self.count = 0

my_arg = Arguments()

def seed_everything(seed):
    #random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 5
seed_everything(seed)


#%% DL part
class ConvLSTM(nn.Module): # convolutional LSTM  
    def __init__(self, spectrum_size, action_size, hidden_size, output_size, num_layers_lstm=2, 
                 num_layers_cnn=5, filters_conv=32, stride=2, kernel_size=51, drop_p_conv=0, drop_p_fc=0):
        super().__init__()       
        self.hidden_size = hidden_size
        self.num_layers_lstm = num_layers_lstm
        self.num_layers_cnn = num_layers_cnn
        self.filters_conv = filters_conv
        self.stride = stride
        self.kernel_size = kernel_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.spectrum_size = spectrum_size
        self.action_size = action_size
        self.output_size = output_size
        self.hidden_size = hidden_size         
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p))
            return conv
        
        # for i2f (input to features)
        block_conv = [] 
        block_conv.append( one_conv(1, self.filters_conv, self.kernel_size, stride=self.stride, drop_p=self.drop_p_conv) )
        self.feature_shape = int( (self.spectrum_size-self.kernel_size)/self.stride +1 )
        for i in range(self.num_layers_cnn-1):
            conv = one_conv(self.filters_conv, self.filters_conv, self.kernel_size, self.stride, drop_p=self.drop_p_conv)
            block_conv.append(conv)
            self.feature_shape = int( (self.feature_shape-self.kernel_size)/self.stride +1)
        self.i2f = nn.Sequential(*block_conv)               # input to features      
        self.ln_features = nn.LayerNorm(self.feature_shape*self.filters_conv+self.action_size)       
        self.lstm = nn.LSTM(self.feature_shape*self.filters_conv+self.action_size, hidden_size, self.num_layers_lstm, batch_first=True)      
        self.ln_lstm = nn.LayerNorm(hidden_size)       
        # i2o
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p_fc)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)           
    def forward(self, inputs, hidden):
        past_obs = inputs.shape[1] 
        #i2f
        conv_part = inputs[:,:, :self.spectrum_size]        # spectra
        fc_part = inputs[:,:, self.spectrum_size:]          # shim actions/values
        features = torch.from_numpy(np.zeros([inputs.shape[0], past_obs, self.feature_shape*self.filters_conv])).float().to(self.DEVICE)
        for k in range(past_obs):                           # convolve each spectrum for its own to keep temporal nature
            features[:,k] = self.i2f(conv_part[:,k].unsqueeze(1)).view(inputs.shape[0],-1)
        #features = self.i2f(conv_part)
        combined = torch.cat((features, fc_part), 2)
        combined = self.ln_features(combined)  
        # LSTM + i2h
        out, (h0,c0) = self.lstm(combined, hidden)
        out = self.ln_lstm(out)    
        #i2o
        x = self.relu(self.fc1(out))
        x = self.drop(x)
        x = self.tanh(self.fc2(x))
        return x, (h0,c0)
    def initHidden(self, batch_size=1):
        return ( torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size), torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size) )

def get_single_model(filename):
    model = ConvLSTM(spectrum_size=2048, action_size=config["nr_shims"], hidden_size=config["hidden_size"], 
                         output_size=config["nr_shims"], num_layers_lstm=2, num_layers_cnn=config["num_layers_cnn"], filters_conv=config["filters_conv"], 
                     stride=config["stride"], kernel_size=config["kernel_size"]) 
    model_state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model
 
#%% Functions
 
# Helper function to save results (+ statistics) to array
def save(spectrum_tmp, prediction_scaled):
    success = (spectrum_tmp/sc_factor).max() > batched_spectra[0,:-config['nr_shims']].max()
    sign_d = np.sign(distortion)
    sign_p = np.sign(-prediction_scaled)
    sign_d[np.where(sign_d==0)[0]] = 1  # count 0 as + sign
    sign_p[np.where(sign_p==0)[0]] = 1
    sign = (sign_d==sign_p).sum()/len(distortion)
    mae = mean_absolute_error(distortion/config['shim_weightings'][:config['nr_shims']]/config['acq_range'],
                       -prediction_scaled/config['shim_weightings'][:config['nr_shims']]/config['acq_range'])
    lw_init_50 = utils_Spinsolve.get_linewidth_Hz(batched_spectra[0,:-config['nr_shims']], sampling_points=32768, bandwidth = 5000)
    lw_shimmed_50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000)    
    lw_init_055 = utils_Spinsolve.get_linewidth_Hz(batched_spectra[0,:-config['nr_shims']], sampling_points=32768, bandwidth = 5000, height=0.9945)
    lw_shimmed_055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp/sc_factor, sampling_points=32768, bandwidth = 5000, height=0.9945)
  
    results_data.append([nr, s, int(0), int(success), sign, int(RANDOM), distortion, prediction_scaled, 
                         lw_init_50.item(), lw_shimmed_50.item(), lw_init_055.item(), lw_shimmed_055.item(), mae, sc_factor])

# Function to perform random step
def random_step(iteration, distortion, batched_spectra, config):
    offset_rand = np.random.uniform(-.5,.5, size=config['nr_shims'])
    offset_rand_scaled = (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int)
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                         np.add(offset_rand_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    if config['full_memory']: full_memory.append([nr, s, sc_factor, fid])
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    batched_spectra = np.append( batched_spectra, np.concatenate((spectrum_tmp/sc_factor, -offset_rand))[np.newaxis,:], axis=0)   
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],spectrum_tmp,'--', label='random{}'.format(iteration), alpha=0.4)
    
    save(spectrum_tmp, (offset_rand*config['acq_range']*config['shim_weightings'][:config['nr_shims']]).astype(int))
    return batched_spectra

# Function to perform predictive step
def step(iteration, distortion, batched_spectra, config):
    rolled_batch = np.roll(batched_spectra, ROLL)
    inputs = torch.tensor(rolled_batch).unsqueeze(0).float()
	(h0,c0) = model.initHidden(batch_size=inputs.shape[0])
	h0, c0 = h0.to(device), c0.to(device)     
	outputs, hidden = model(inputs, (h0,c0))       	# make prediction  
	prediction = outputs[:,-1]						# take last pred
	prediction = prediction.detach().numpy()[0]
    
    #clip to prevent bad currents
    prediction_scaled = np.clip(-10000, (prediction*config['shim_weightings'][:config['nr_shims']]*50).astype(int), 10000)
             
    if input_args.verbose >= 1: 
        print('artificial distortion (x,y,z,z2): ', distortion)
        print('predicted correction (x,y,z,z2): ', prediction_scaled)

    # take prediction as step
    xaxis, spectrum_tmp, fid, shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, 
                            np.add(prediction_scaled,distortion), True,True, verbose=(input_args.verbose>1))
    if config['full_memory']: full_memory.append([nr, s, sc_factor, fid])
    my_arg.count += 1
    spectrum_tmp = spectrum_tmp[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
    if input_args.verbose == 2: print('shims: ', shims)
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp,
            label='$u({})$'.format(iteration))
    
    batched_spectra = np.append( batched_spectra, np.concatenate((spectrum_tmp/sc_factor, -prediction))[np.newaxis,:], axis=0)
    
    save(spectrum_tmp, prediction_scaled)
    return batched_spectra, prediction_scaled
    
#%%       
com = utils_Spinsolve.init( verbose=(input_args.verbose>0))

# VARIABLES
STEPS_RAND = 7
STEPS_REAL = 2

# loop over rand steps to measure influence
# "TAB" code if used
#for STEPS_RAND in range(6,11):

global spectra_memory       # memory for 2048p ROI of spectra
spectra_memory = []
global full_memory          # big memory for whole fid (not only ROI)
full_memory = [] 
global results_data
results_data = []


# Get best spectrum
xaxis, best_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, np.zeros(config['nr_shims']),
                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
my_arg.count += 1
spectrum_tmp = best_spectrum[config["ROI_min"]:config["ROI_min"]+2048]/config["max_data"]
plt.figure()
plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048], spectrum_tmp)
plt.title('Best spectrum with {} shims'.format(config['nr_shims']))
plt.xlim(PLT_X)             # crop for plotting♠
plt.savefig(DATAPATH + '/eDR/img_eDR_{}_best.png'.format(config["sample"]))
plt.show()

# store best before experiment
best50 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp, sampling_points=32768, bandwidth = 5000)   
best055 = utils_Spinsolve.get_linewidth_Hz(spectrum_tmp, sampling_points=32768, bandwidth = 5000, height=0.9945)
columns_data = ['lw50', 'lw055','Spectrum']
df_spectra = pd.DataFrame([[best50.item(), best055.item(), list(spectrum_tmp)]], columns=columns_data)   

print(best50)
df_spectra.to_excel(DATAPATH + '/eDR/best_spectrum_eDR_id{}_ps{}_{}_r{}s{}_{}.xlsx'.format(
    initial_config["id"],('Y' if config['postshim'] else 'N'),config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))


# loop over all random distortions and track performance
for nr, distortion in enumerate(random_distortions):

    model = get_single_model(DATAPATH+'/eDR/models/model_DR_Z2_{}_id{}.pt'.format(config['set'],config["id"]))

    xaxis, initial_spectrum, fid, ref_shims = utils_Spinsolve.setShimsAndRunV3(com, my_arg.count, distortion,
                                                return_shimvalues=True, return_fid=True, verbose=(input_args.verbose>1))
    if config['full_memory']: full_memory.append([nr, 0, 1, fid])
    my_arg.count += 1
    linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
    initial_spectrum = initial_spectrum[config["ROI_min"]:config["ROI_min"]+2048] / config["max_data"] # scale to dataset        
    if config['scale_sample']: sc_factor = initial_spectrum.max()
    else: sc_factor = 1
    
    plt.figure()
    plt.plot(xaxis[config["ROI_min"]:config["ROI_min"]+2048],initial_spectrum, label='initial')
    
    batched_spectra = np.concatenate((initial_spectrum/sc_factor, np.zeros(config['nr_shims'])))[np.newaxis,:] # first 
    
    # random steps
    for s in range(1, STEPS_RAND+1):
        RANDOM = True
        batched_spectra = random_step(s, distortion, batched_spectra, config)

    # real steps
    for s in range(STEPS_RAND+1,STEPS_REAL+STEPS_RAND+1):
        RANDOM = False
        batched_spectra, prediction_scaled = step(s, distortion, batched_spectra, config)

    plt.xlim(PLT_X) # crop for plotting
    plt.legend()    
    plt.title('Shimming with eDR')
    plt.ylabel("Signal [a.u.]")
    plt.xlabel("Frequency [Hz]")
    plt.savefig(DATAPATH + '/eDR/plots/img_eDR_{}_id{}_ps{}_r{}s{}_{}.png'.format(
        config["sample"],initial_config["id"],('Y' if config['postshim'] else 'N'),STEPS_RAND,STEPS_REAL,nr))
    #plt.show()
    
    random_array = np.append(np.append(np.array([-1]),np.repeat(1,STEPS_RAND)), np.repeat(0,STEPS_REAL))
    for i, tmp in enumerate(batched_spectra): 
        spectra_memory.append([nr, i, random_array[i], sc_factor, list(tmp[:2048]), list(tmp[-config['nr_shims']:])])


# Convert results to pd df
columns_data = ['Nr', 'Step', 'Inverted', 'Success', 'Sign', 'Random', 'Distortion', 'Prediction', 
                'lw50_init', 'lw50_step', 'lw55_init', 'lw55_step', 'MAE', 'ScalingFactor']
df = pd.DataFrame(results_data, columns=columns_data)
df.to_excel(DATAPATH + '/eDR/results_eDR_id{}_ps{}_{}_r{}s{}_{}.xlsx'.format(
        initial_config["id"],('Y' if config['postshim'] else 'N'),config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

    
# print results
print("Success rate: ", df.loc[df['Step']==STEPS_REAL+STEPS_RAND-1]['Success'].mean())
print("Correct prediction rate: ", round( df.loc[df['Step']==STEPS_REAL+STEPS_RAND-1]['Sign'].mean() , 3))
print("Averaged MAE: {} +/- {}".format(round(df.loc[df['Step']==STEPS_REAL+STEPS_RAND-1]['MAE'].mean(),2), round(df.loc[df['Step']==STEPS_REAL+STEPS_RAND-1]['MAE'].mean(),2)) )

# convert memory to pd df
# Excel cell limit is 32767 --> Store as pickle
columns_data = ['Nr', 'Step', 'Random', 'ScalingFactor', 'Spectrum', 'ShimOffsets']
df_spectra = pd.DataFrame(spectra_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/eDR/spectra_memory_eDR_id{}_ps{}_{}_r{}s{}_{}.pickle'.format(
    initial_config["id"],('Y' if config['postshim'] else 'N'),config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

# convert memory to pd df
columns_data = ['Nr', 'Step', 'ScalingFactor', 'Spectrum']
df_spectra = pd.DataFrame(full_memory, columns=columns_data)    
df_spectra.to_pickle(DATAPATH + '/eDR/spectra_memoryFULL_eDR_id{}_ps{}_{}_r{}s{}_{}.pickle'.format(
    initial_config["id"],('Y' if config['postshim'] else 'N'),config["sample"],STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')))

import json
with open(DATAPATH + '/eDR/config_eDR_id{}_ps{}_{}_r{}s{}_{}.json'.format(
    initial_config["id"],('Y' if config['postshim'] else 'N'),config["sample"],
    STEPS_RAND,STEPS_REAL,datetime.today().strftime('%Y-%m-%d-%HH-%mm')), 'w') as f:
    json.dump(config,f)

# run automated shim to guarantee same starting point for next iteration
if config['postshim']:
    print('Post-shimming #1. Please wait...')
    com.RunProspaMacro(b'QuickShim()')
    com.RunProspaMacro(b'gExpt->runExperiment()')
    print('Post-shimming #2. Please wait...')
    com.RunProspaMacro(b'QuickShim()')
    com.RunProspaMacro(b'gExpt->runExperiment()')
    

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))
