######### - Library and models- #########

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

from src.models.cnns_1 import modello1 as model #Choose the model to train and test 

######### - Loading data - #########

data = np.load("data/dataset_ground_states_N_18_istanze_100mila.npz", mmap_mode='r')

gs = data["ground_state_energy"]
pot_energy = data["potenergy"]
density = data["density"]
potential = data["potential"]

kinetic_energy = gs - pot_energy

#print(f'ground state energy {gs.shape}, potential energy shape {pot_energy.shape}, density shape {density.shape}, potential shape {potential.shape}')


######### - Subdivision of data in 70% training, 15% validation and 15% test set - #########

#Training set
train_KE = kinetic_energy[:70000] 
train_density = density[:70000]

#Validation set
val_KE = kinetic_energy[70000:85000]
val_density = density[70000:85000]

#Test set
test_KE = kinetic_energy[85000:]
test_density = density[85000:]

######### - Tensor creation with right dimensions - #########
train_KE, test_KE, train_density, test_density, val_density, val_KE = map(lambda x: torch.from_numpy(x).float(), (train_KE, test_KE, train_density, test_density,  val_density, val_KE ) )
train_density = train_density.view(-1,1,18,18,18) 
test_density = test_density.view(-1,1,18,18,18)
val_density = val_density.view(-1,1,18,18,18)

train_KE = train_KE.view(-1,1)
test_KE = test_KE.view(-1,1)
val_KE = val_KE.view(-1,1)

######### - Dataset and dataloader - #########

#Training
train_ds = TensorDataset(train_density, train_KE)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

#Validation
val_ds = TensorDataset(val_density, val_KE)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=True)

#Test
test_ds = TensorDataset(test_density, test_KE)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)



funzionale_energia = model().to(device) # Load the model to the GPU


n_epoch = 10 #Number of epoch
learning_rate = 1e-4 
allenamento = f"{str(funzionale_energia.__class__.__name__)}" # Name of the model in string
loss_function = nn.MSELoss() #Use Mean Squared Error as the loss function.
optimizer = optim.Adam(funzionale_energia.parameters(), lr = learning_rate)  #Use the Adam optimization algorithm
#dim = 60


#Logging and saving loss values of training and validation set and time of training
log_train_loss = []
log_val_loss = []

log_train_R2 = []
log_val_R2 = []

log_train_file = open(f"results/logs/{allenamento}_epoche={n_epoch}_lr={learning_rate}_1ferm.txt", "w")
log_train_file.write('EPOCA\tTRAIN_LOSS\tVAL_LOSS\tR2_TRAIN\tR2_VAL\n')

start_time = time.time()

for epoch in range(n_epoch):

    train_pred = []
    train_target = []
    #Training

    total_loss = 0.0
    funzionale_energia.train()

    for i, data in enumerate(train_dl, 0):
        density = data[0].to(device)
        KE = data[1].to(device)
        
        optimizer.zero_grad()
        output = funzionale_energia(density)
        
        #Compute loss function and update parameters
        loss = loss_function(output, KE)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()


        output = output.squeeze()
        train_pred.append(output.detach().cpu().numpy())
        train_target.append(KE.detach().cpu().numpy())

    avg_train_loss = total_loss / len(train_dl)
    log_train_loss.append(avg_train_loss)

    train_pred = np.concatenate(train_pred)
    train_target = np.concatenate(train_target)

    total_loss = 0.0

    #Validation
    
    val_pred = []
    val_target = []

    funzionale_energia.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dl, 0):
            density = data[0].to(device)
            KE = data[1].to(device)
            
            output = funzionale_energia(density)
            

            #Compute loss function
            loss = loss_function(output, KE)
            total_loss += loss.item()

            output = output.squeeze()
            val_pred.append(output.detach().cpu().numpy())
            val_target.append(KE.detach().cpu().numpy())


        avg_val_loss = total_loss / len(val_dl)
        log_val_loss.append(avg_val_loss)
        
        val_pred = np.concatenate(val_pred)
        val_target = np.concatenate(val_target)

        #Compute R^2 on train set
        R2_train = r2_score(train_target,train_pred)
        log_train_R2.append(R2_train)

        #Compute R^2 on validation set
        R2_val = r2_score(val_target,val_pred)
        log_val_R2.append(R2_val)

        print(f'{[epoch+1]} loss su train: {avg_train_loss}, loss su val: {avg_val_loss}, R^2 su train: {R2_train}, R^2 su val: {R2_val}')

    log_train_file.write(f'{[epoch+1]}\t{avg_train_loss}\t{avg_val_loss}\t{R2_train}\t{R2_val}\n')

end_time = time.time()
time_elapsed = end_time - start_time
print(f'Tempo impiegato: {time_elapsed} s')
log_train_file.write(f'{time_elapsed}')
log_train_file.close()
torch.save(funzionale_energia.state_dict(), f"results/weight/{allenamento}_epoche={n_epoch}_lr={learning_rate}.pth")


    