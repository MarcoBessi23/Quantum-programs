import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi 
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset


#### PAULI GATE 
# qc.x(0) applicato al qubit 0
#### PAULI ROTATING GATE
# qc.rx(theta, 0) applicato al qubit 0
#### HADAMARD GATE
# qc.h() applicato al qubit 0
#### CONTROLLED Y GATE
# qc.cy(control_qubit, target_qubit)
#### MULTI CONTROLLED Z
# qc.mcp(pi, control_qubits, target)
#### DISCRETE PRIMITIVE
#




##### AMPLITUDE ENCODER

#vector = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])  # Vettore da codificare (deve essere normalizzato)
#
#qc = QuantumCircuit(2)  # Per 4 stati servono 2 qubit
#qc.append(StatePreparation(vector), [0, 1])
#
##qc.draw('mpl')
#print(qc.draw(output='text'))




transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

train_dataset_filtered = [(img, label) for img, label in train_dataset if label == 0 or label == 1]
test_dataset_filtered = [(img, label) for img, label in test_dataset if label == 0 or label == 1]

train_images = []
train_labels = []
test_images = []
test_labels = []

for img, label in train_dataset_filtered:
    if len(train_images) < 500 and label == 0:
        train_images.append(img.view(-1).numpy())  # Appiattisci l'immagine
        train_labels.append(label)
    elif len(train_images) < 1000 and label == 1:
        train_images.append(img.view(-1).numpy())  # Appiattisci l'immagine
        train_labels.append(label)

for img, label in test_dataset_filtered:
    if len(test_images) < 50 and label == 0:
        test_images.append(img.view(-1).numpy())  # Appiattisci l'immagine
        test_labels.append(label)
    elif len(test_images) < 100 and label == 1:
        test_images.append(img.view(-1).numpy())  # Appiattisci l'immagine
        test_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(train_images.shape, train_labels.shape)  # Dovrebbe essere (1000, 784), (1000,)
print(test_images.shape, test_labels.shape)    # Dovrebbe essere (100, 784), (100,)



##IMPORTANT TO USE PCA
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

pca = PCA(n_components=8)
train_images_pca = pca.fit_transform(train_images_scaled)
test_images_pca = pca.transform(test_images_scaled)

print(train_images_pca.shape)
print(test_images_pca.shape)


##BUILD THE NEURAL NETWORK

from qiskit.circuit import Parameter
import qiskit.circuit.library as qulib



def amplitude_encoding(input_vector:np.array):
    '''
    Amplitude encoding for a vecto of size 8, it takes |000> and 
    the input vector and brings it in the desired encoded state.
    '''
    
    norm = np.linalg.norm(input_vector)
    quantum_state = input_vector/norm
    
    def calculate_theta(values):
        """Procedure that returns the angle that for RY"""
        norm_factor = np.linalg.norm(values)
        return 2 * np.arctan(np.sqrt(sum(values[len(values)//2:]**2) / sum(values[:len(values)//2]**2)))

    theta_1 = calculate_theta(quantum_state)      
    theta_2 = calculate_theta(quantum_state[:4])  
    theta_3 = calculate_theta(quantum_state[4:])  
    theta_4 = calculate_theta(quantum_state[:2])  
    theta_5 = calculate_theta(quantum_state[2:4])
    theta_6 = calculate_theta(quantum_state[4:6])
    theta_7 = calculate_theta(quantum_state[6:])


    qc = QuantumCircuit(4)
    # Applichiamo la decomposizione ad albero
    qc.ry(theta_1, 1)  # separate indices [0,1,2,3] from [4,5,6,7] 
    qc.cx(1, 2)
    qc.ry(theta_2, 2)  # Secondo livello: separa 2 gruppi da 2 nei primi 4
    qc.ry(theta_3, 2)  # Secondo livello: separa 2 gruppi da 2 negli ultimi 4
    qc.cx(2, 3)
    qc.ry(theta_4, 3)  # Terzo livello: separa i primi 2
    qc.ry(theta_5, 3)  # Terzo livello: separa i secondi 2
    qc.ry(theta_6, 3)  # Terzo livello: separa i terzi 2
    qc.ry(theta_7, 3)  # Terzo livello: separa gli ultimi 2

    return qc



def flexible_oracle(qc, theta):
    
    #|000>
    qc.rx(theta[0], 0)
    control0 = qulib.C3XGate(3, ctrl_state='1000')    
    qc.append(control0)
    
    #|001>
    qc.rx(theta[1], 0)
    control1 = qulib.C3XGate(3, ctrl_state='1001')    
    qc.append(control1)

    #|010>
    qc.rx(theta[2], 0)
    control2 = qulib.C3XGate(3, ctrl_state='1010')    
    qc.append(control2)
    
    #|011>
    qc.rx(theta[3], 0)
    control3 = qulib.C3XGate(3, ctrl_state='1011')    
    qc.append(control3)
    
    #|100>
    qc.rx(theta[4], 0)
    control4 = qulib.C3XGate(3, ctrl_state='1100')    
    qc.append(control4)

    #|101>
    qc.rx(theta[5],0)
    control5 = qulib.C3XGate(3, ctrl_state='1101')    
    qc.append(control5)
    
    #|110>
    qc.rx(theta[6],0)
    control6 = qulib.C3XGate(3, ctrl_state='1110')    
    qc.append(control6)

    #|111>
    qc.rx(theta[7],0)
    control7 = qulib.C3XGate(3, ctrl_state='1111')    
    qc.append(control7)

    return qc

###COSTRUCT THE MULTICONTROLL GATE
zcir = QuantumCircuit(1)
zcir.z(0)
zgate = zcir.to_gate(label='z').control(3, ctrl_state= '111')


def adaptive_diffusion(qc, psi):
    ''' 
    psi is the training parameter
    '''
    qc.h([1,2,3])
    qc.cry(psi[0], 1, 2)
    qc.cry(psi[1], 2, 3)
    qc.cry(psi[2], 3, 1)
    ###append the multicontrol Z gate
    qc.append(zgate[0,1,2,3])
    qc.cry(psi[3], 3, 1)
    qc.cry(psi[4], 2, 3)
    qc.cry(psi[5], 1, 2)

    qc.h([1,2,3])

    return qc


def GQHAN(input, theta, psi):
    qc = amplitude_encoding(input)
    qc = flexible_oracle(qc, theta)
    qc = adaptive_diffusion(qc, psi)

    return qc

from qiskit.circuit import ParameterVector
parameters_FO = ParameterVector('theta', length = 8)
parameters_ADO = ParameterVector('psi', length = 6)




from qiskit import  transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

def cost_function(params, input_data, labels):
    backend = AerSimulator()
    shots = 512
    total_cost = 0

    for i, x in enumerate(input_data):
        print(f'image number {i}')
        qc = GQHAN(x, params[:8], params[8:])
        
        qc.measure_all()  # Misura tutti i qubit
        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc).result()
        counts = result.get_counts(transpiled_qc)

        p1 = counts.get('1', 0) / shots  

        y = labels[i]

        # Loss function (Cross-Entropy)
        total_cost += -y * np.log(p1 + 1e-9) - (1 - y) * np.log(1 - p1 + 1e-9)
    
    return total_cost / len(input_data)


# Inizializzazione casuale dei parametri
init_params = np.random.uniform(0, 2*np.pi, 14)  # 8 per theta, 6 per psi

# Ottimizzazione con COBYLA
opt_result = minimize(cost_function, init_params, args=(train_images_pca, train_labels), method='COBYLA')

# Parametri ottimizzati
trained_params = opt_result.x
print("Parametri ottimizzati:", trained_params)
