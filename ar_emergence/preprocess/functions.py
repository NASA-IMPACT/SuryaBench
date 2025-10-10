# functions to be used in pipeline

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Splitting Image Function
def split_image(image,size): #Splits a given image of size 512 by 512 into 9 equal-size square pieces. Returns a list of 9 PIL Image objects.
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    piece_size = width//size
    pieces = []
    for i in range(size):
        for j in range(size):
            x = j * piece_size
            y = i * piece_size
            im = Image.fromarray(image)
            piece = im.crop((x, y, x+piece_size, y+piece_size))
            pieces.append(np.array(piece))
    return np.array(pieces)

# Calculate split image mean
def get_piece_means(input_cube,size): 
    parts = size**2
    means_timeline = np.zeros((np.shape(input_cube)[0],parts))
    for frame_num in range(0,np.shape(input_cube)[0]): # for every frame
        pieces = split_image(input_cube[frame_num,:,:],size)
        means_timeline[frame_num,:] = np.array([np.mean(array) for array in pieces])
    return np.transpose(means_timeline)

# Calculate DTW
def dtws(size,pm_means): #Should we take in account the neighboroung time points?
    dtw = np.zeros(np.shape(pm_means))
    dtwA = pm_means[:size,:]
    dtwB = pm_means[-size:,:]
    for i in range(0,size):
        #print(dtw_weights(size)[-(i+1)],dtw_weights(size)[i])
        dtw[i*size:(i+1)*size,:] = dtw_weights(size)[-(i+1)]*dtwA + dtw_weights(size)[i]*dtwB
    return dtw

# Get weighting for each row of grid
def dtw_weights(size):
    if size % 2 == 0: 
        my_list = np.arange(0,size+1)
        my_list = np.delete(my_list,int(size/2))/size
        print(my_list)
        sys.exit()
    else: 
        my_list = np.linspace(0, 1, num=size+1)
        average = (my_list[(size+1)//2 - 1] + my_list[(size+1)//2])/2 # Calculate the average of the two middle elements
        my_list[(size+1)//2-1 : (size+1)//2+1] = [average] # Replace the two middle elements with the calculated average
        my_list = np.delete(my_list,int(size/2))
    return my_list

# Straighten lines
def straighten(dtw_dist,old_line):
    new_line = 1
    return new_line

# Calculate derivatives
def derivative():
    x = 1
    return x

def scale_vid(x, pos, new_min, new_max): # define the mapping function
    return '{:.2f}'.format(new_min + ((x - 0) / (512 - 0)) * (new_max - new_min))

def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', length=50, fill='#', empty='-', end_line='\r'):
    progress = float(iteration) / float(total)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {iteration}/{total} {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write(end_line)
        sys.stdout.flush()

# Functions for comparing frames in corrupt due to eclipse AR11726

def calculate_frame_difference_metric(framez):
    num_frames = framez.shape[0]
    frame_diffs = []
    for i in range(1, num_frames):
        diff = np.mean(np.abs(framez[i] - framez[i - 1]))
        frame_diffs.append(diff)
    return frame_diffs

def plot_frame_difference_metric(frame_diffs,cor_file):
    zero_start = None
    zero_end = None
    in_zeros = False
    for i, diff in enumerate(frame_diffs):
        if not in_zeros and diff == 0:
            in_zeros = True
            zero_start = i
        elif in_zeros and diff != 0:
            in_zeros = False
            zero_end = i - 1
            break
    if zero_start is not None and zero_end is not None:
        print(f"The frame_diffs array contains zeros from index {zero_start} to {zero_end} before becoming non-zero values again.")
    elif zero_start is not None:
        print(f"The frame_diffs array contains zeros starting from index {zero_start}, but non-zero values are not encountered again.")
    else:
        print("The frame_diffs array does not contain consecutive zeros.")
    plt.plot(frame_diffs)
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Difference Metric')
    plt.title(f'Frame Difference Metric vs. Frame Index for {cor_file}')
    plt.show()


##### Sept 18th and later


def lstm_ready(source,AR,tile,size):
    # Read AR and create lstm ready data
    power_maps = np.load('/nobackup/skasapis/AR{}/power_maps_{}{}.npz'.format(AR,source,AR),allow_pickle=True) 
    intensities = np.load('/nobackup/skasapis/AR{}/intensities{}.npz'.format(AR,AR),allow_pickle=True) 
    power_maps23 = get_piece_means(power_maps['arr_0'],size)
    power_maps34 = get_piece_means(power_maps['arr_1'],size)
    power_maps45 = get_piece_means(power_maps['arr_2'],size)
    power_maps56 = get_piece_means(power_maps['arr_3'],size)
    intensities = get_piece_means(intensities['arr_0'],size)
    power_maps23 = power_maps23 - dtws(size,power_maps23) # Flatten arrays
    power_maps34 = power_maps34 - dtws(size,power_maps34)
    power_maps45 = power_maps45 - dtws(size,power_maps45)
    power_maps56 = power_maps56 - dtws(size,power_maps56)
    intensities = intensities - dtws(size,intensities)
    power_maps23 = power_maps23[size:-size, :] # Trim array to get rid of top and bottom 0 tiles
    power_maps34 = power_maps34[size:-size, :]
    power_maps45 = power_maps45[size:-size, :]
    power_maps56 = power_maps56[size:-size, :]
    intensities = intensities[size:-size, :] 
    stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1)
    final_maps = np.transpose(stacked_maps, axes=(2, 1, 0))
    final_ints = np.transpose(intensities, axes=(1,0))
    
    all_power_maps = final_maps[:,:,tile]
    all_intensities = final_ints[:,tile]

    mm = MinMaxScaler()
    ss = StandardScaler()

    X_trans = ss.fit_transform(all_power_maps)
    y_trans = mm.fit_transform(all_intensities.reshape(-1, 1))

    X_ss, y_mm = split_sequences(X_trans, y_trans, 10, 5)

    #print(X_ss.shape, y_mm.shape)
    #print('check agreement:')
    #print(y_mm[0])
    #print(y_trans[9:14].squeeze(1))

    return X_ss, y_mm

def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train, X_test, y_test):
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 10 == 0: print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item())) 

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out

# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)