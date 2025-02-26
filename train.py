import os
import numpy as np
from qiskit.circuit import ParameterVector, Parameter
from qiskit_algorithms.gradients import ParamShiftSamplerGradient, ParamShiftEstimatorGradient
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.primitives import Estimator
from data_preprocessing import PCA_data
from quantum_NeuralNet import GQHAN

train_images_pca, test_images_pca, train_labels, test_labels = PCA_data()

def cost_function(params, batch, labels):
    '''
    Calculate loss function as shown in https://arxiv.org/abs/2401.14089    
    '''
    total_cost = 0
    for i, x in enumerate(batch):
        
        qc = GQHAN(x)
        z_op = Pauli("ZIII")
        estimator = Estimator()
        job = estimator.run([qc], [z_op], [params])
        expectation_value = job.result().values[0]
        print(expectation_value)
        
        y = labels[i]
        #Binary Cross-Entropy
        total_cost += (y-np.sign(expectation_value))**2
        
    return total_cost / len(batch)

def compute_gradient(params:np.ndarray, batch):
    '''
    compute gradient of the quantum part of the loss
    '''

    estimator = Estimator()
    z_op = Pauli("ZIII")
    grad = np.zeros_like(params)
    print('Gradient calculation')
    for _, x in enumerate(batch):
        qc = GQHAN(x)
        gradient = ParamShiftEstimatorGradient(estimator)
        pse_grad_result = gradient.run(qc, z_op, [params.tolist()]).result().gradients
        g = np.array(pse_grad_result)[0]
        grad += g
        
    return grad/len(batch)

def nesterov(train_data, train_labels, init_params, learning_rate=0.09, momentum=0.9, epochs=4, batch_size=30):
    params = init_params
    velocity = np.zeros_like(params)

    for _ in range(epochs):
        # Shuffle dataset
        indices = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[indices], train_labels[indices]
        loss_values = []

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            batch_labels = train_labels[i : i + batch_size]
            print(np.shape(batch))
            
            lookahead_params = params - momentum * velocity
    
            ##COMPUTE GRADIENT OF LOSS FUNCTION WITH RESPECT TO QC PARAMETER, FIRST CONSIDER THE CHAIN RULE
            ##THEN USE PARAMETER SHIFT RULE FOR THE QUANTUM CIRCUIT
            cost = 2*cost_function(lookahead_params, batch, batch_labels)
            gradient = compute_gradient(lookahead_params, batch)
            loss_gradient = cost*gradient

            velocity = momentum * velocity + learning_rate * loss_gradient
            params -= velocity

            print(f'loss at iteration {i} is {cost/2}')
            loss_values.append(cost/2)
        
    return params, loss_values

np.random.seed(42)
init_params = np.random.uniform(0, 2*np.pi, 14)
optimized_params, loss = nesterov(train_images_pca, train_labels, init_params)

save_dir = os.makedirs('parameters', exist_ok=True)
save_path = os.path.join(os.getcwd(), save_dir, "optimized_params.npy")
np.save(save_path, optimized_params)