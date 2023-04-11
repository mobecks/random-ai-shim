import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

#https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/44
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('tanh'))
        torch.nn.init.constant_(m.bias, 0)


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
        
        self.lstm = nn.LSTM(self.feature_shape*self.filters_conv+self.action_size, hidden_size, self.num_layers_lstm,
                            batch_first=True)
        
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
