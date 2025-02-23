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

parameter = Parameter('theta')
qc = QuantumCircuit(4)

