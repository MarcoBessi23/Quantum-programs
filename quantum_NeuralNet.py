import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit.circuit import ParameterVector #faster than defining 14 different parameters
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize




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
        train_images.append(img.view(-1).numpy())
        train_labels.append(label)
    elif len(train_images) < 1000 and label == 1:
        train_images.append(img.view(-1).numpy())
        train_labels.append(label)

for img, label in test_dataset_filtered:
    if len(test_images) < 50 and label == 0:
        test_images.append(img.view(-1).numpy())
        test_labels.append(label)
    elif len(test_images) < 100 and label == 1:
        test_images.append(img.view(-1).numpy())
        test_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(train_images.shape, train_labels.shape)  #(1000, 784), (1000,)
print(test_images.shape, test_labels.shape)    #(100, 784), (100,)



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


def amplitude_encoding(input_vector:np.array):
    '''
    Amplitude encoding for a vecto of size 8, it takes |000> and 
    the input vector and brings it in the desired encoded state.
    '''
    
    norm = np.linalg.norm(input_vector)
    quantum_state = input_vector/norm
    
    def calculate_theta(values):
        """Procedure that returns the angle for RY"""
        norm_factor = np.linalg.norm(values)
        return 2 * np.arctan(np.sqrt(sum(values[len(values)//2:]**2) / sum(values[:len(values)//2]**2)))


    ###HERE I WANT TO APPLY THE ROTATION NECESSARY TO GET FROM |000> TO THE ENCODED STATE I USE THE TREE METHOD
    theta_1 = calculate_theta(quantum_state)      
    theta_2 = calculate_theta(quantum_state[:4])  
    theta_3 = calculate_theta(quantum_state[4:])  
    theta_4 = calculate_theta(quantum_state[:2])  
    theta_5 = calculate_theta(quantum_state[2:4])
    theta_6 = calculate_theta(quantum_state[4:6])
    theta_7 = calculate_theta(quantum_state[6:])

    qc = QuantumCircuit(4, 3)
    qc.ry(theta_1, 1) 
    qc.cx(1, 2)
    qc.ry(theta_2, 2)
    qc.ry(theta_3, 2)
    qc.cx(2, 3)
    qc.ry(theta_4, 3)
    qc.ry(theta_5, 3)
    qc.ry(theta_6, 3)
    qc.ry(theta_7, 3)

    return qc

def control_lambda(control_qubits:str):
    xcir = QuantumCircuit(1)
    xcir.x(0)
    xgate = xcir.to_gate(label='X').control(3, ctrl_state= control_qubits)
    
    return xgate    

def flexible_oracle(qc, theta):
    
    #|000>
    qc.rx(theta[0], 0)
    qc.x(3)#put third bit in 1, same idea as for grover exercise in grover.py 
    qc.h(3)
    control0 = control_lambda('001')    
    qc.append(control0, [0,1,2,3])
    qc.h(3)
    qc.x(3)

    #|001>
    qc.rx(theta[1], 0)
    qc.h(3)
    qc.append(control0, [0,1,2,3])
    qc.h(3)

    #|010>
    qc.rx(theta[2], 0)
    
    qc.x(3)
    qc.h(3)
    control2 = control_lambda('101')    
    qc.append(control2, [0,1,2,3])
    qc.h(3)
    qc.x(3)
    
    #|011>
    qc.rx(theta[3], 0)
    qc.h(3)
    qc.append(control2, [0,1,2,3])
    qc.h(3)
    
    #|100>
    qc.rx(theta[4], 0)
    qc.x(3)
    qc.h(3)
    control3 = control_lambda('011')    
    qc.append(control3, [0,1,2,3])
    qc.h(3)
    qc.x(3)

    #|101>
    qc.rx(theta[5],0)
    qc.h(3)
    qc.append(control3, [0,1,2,3])
    qc.h(3)

    #|110>
    qc.rx(theta[6],0)
    qc.x(3)
    qc.h(3)
    control4 = control_lambda('111')
    qc.append(control4, [0,1,2,3])
    qc.h(3)
    qc.x(3)

    #|111>
    qc.rx(theta[7],0)
    qc.h(3)
    qc.append(control4, [0,1,2,3])
    qc.h(3)

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
    qc.append(zgate, [0,1,2,3])
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

parameters_FO = ParameterVector('theta', length = 8)
parameters_ADO = ParameterVector('psi', length = 6)


def cost_function(params, input_data, labels):
    
    backend = QasmSimulator()
    total_cost = 0

    for i, x in enumerate(input_data):
        print(f'image number {i}')
        qc = GQHAN(x, params[:8], params[8:])
        
        qc.measure([1, 2, 3], [0, 1, 2])
        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc).result()
        counts = result.get_counts(transpiled_qc)
        
        ##Uncomment to get an Idea of the results
        #print(counts)
        #print(type(counts))

        shots = np.sum(np.array(list(counts.values())))
        third_qubit = np.array([counts[key] for key in counts if key.endswith('1')])
        p1 = np.sum(third_qubit) / shots
        print(p1)
        y = labels[i]
        
        #Binary Cross-Entropy
        total_cost += (y-p1)**2
    
    return total_cost / len(input_data)

init_params = np.random.uniform(0, 2*np.pi, 14)  # 8 per theta, 6 per psi
v = cost_function(init_params, train_images_pca, train_labels)


#opt_result = minimize(cost_function, init_params, args=(train_images_pca, train_labels), method='COBYLA')

# Parametri ottimizzati
#trained_params = opt_result.x
#print("Optimized Parameters:", trained_params)
#
#
#def predict(params, input_data):
#    backend = AerSimulator()
#    shots = 512
#    predictions = []
#
#    for i, x in enumerate(input_data):
#        qc = GQHAN(x, params[:8], params[8:])
#        qc.measure_all()
#        
#        transpiled_qc = transpile(qc, backend)
#        result = backend.run(transpiled_qc).result()
#        counts = result.get_counts(transpiled_qc)
#
#        p1 = counts.get('1', 0) / shots
#
#        #ARGMAX OF PROBABILITY
#        prediction = 1 if p1 > 0.5 else 0
#        predictions.append(prediction)
#
#    return np.array(predictions)
#
#test_predictions = predict(trained_params, test_images_pca)
#accuracy = np.mean(test_predictions == test_labels)
#print(f"Test Accuracy : {accuracy * 100:.2f}%")