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
train_labels[train_labels == 0] = -1
test_labels[test_labels == 0] = -1



z_op = Pauli("ZIII")
print(z_op)

def accuracy(params, input_data, labels):
    y_pred = []
    for i, x in enumerate(input_data):
        
        qc = GQHAN(x)
        z_op = Pauli("ZIII") ###see https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli
        estimator = Estimator()
        job = estimator.run([qc], [z_op], [params])
        expectation_value = job.result().values[0]
        y_pred.append(np.sign(expectation_value))
    
    acc = np.sum(labels == y_pred)/len(labels)
    return acc



def cost_function(params, batch, labels):
    '''
    Calculate loss function as shown in https://arxiv.org/abs/2401.14089    
    '''
    first_derivative = 0
    total_cost = 0
    for i, x in enumerate(batch):
        
        qc = GQHAN(x)
        z_op = Pauli("ZIII")
        estimator = Estimator()
        job = estimator.run([qc], [z_op], [params])
        expectation_value = job.result().values[0]
        
        y = labels[i]
        #Binary Cross-Entropy
        total_cost += (y-np.sign(expectation_value))**2
        first_derivative += (y-np.sign(expectation_value))
    
    return total_cost / len(batch), -2 * first_derivative / len(batch)

#def dL_df(params, batch, labels):
#    '''
#    Calculate first part of gradient    
#    '''
#    total_cost = 0
#    for i, x in enumerate(batch):
#        
#        qc = GQHAN(x)
#        z_op = Pauli("ZIII")
#        estimator = Estimator()
#        job = estimator.run([qc], [z_op], [params])
#        expectation_value = job.result().values[0]
#        
#        y = labels[i]
#        #Binary Cross-Entropy
#        total_cost += (y-np.sign(expectation_value))
#        
#    return -2 * total_cost / len(batch)
#

def compute_gradient(params:np.ndarray, batch):
    '''
    compute gradient of the quantum part of the loss using parameter shift
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

    for epoch in range(epochs):
        print(f'EPOCH NUMBER {epoch}')
        indices = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[indices], train_labels[indices]
        loss_values = []

        for i in range(0, len(train_data), batch_size):
            print(i)
            batch = train_data[i : i + batch_size]
            batch_labels = train_labels[i : i + batch_size]            
            lookahead_params = params - momentum * velocity

            ##COMPUTE GRADIENT OF LOSS FUNCTION WITH RESPECT TO QC PARAMETER, FIRST CONSIDER THE CHAIN RULE
            ##THEN USE PARAMETER SHIFT RULE FOR THE QUANTUM CIRCUIT
            loss, dL_df = cost_function(lookahead_params, batch, batch_labels)
            gradient = compute_gradient(lookahead_params, batch)
            loss_gradient = dL_df*gradient

            velocity = momentum * velocity + learning_rate * loss_gradient
            params -= velocity
            
            if i%60 == 0:
                print(f'loss at iteration {i} is {loss:.4f}')
                loss_values.append(loss)
        print('Evaluating test accuracy')
        acc = accuracy(params, test_images_pca, test_labels)
        print(f'test accuracy at epoch {epoch}: {acc:.2f}')

    return params, loss_values, acc

np.random.seed(42)
init_params = np.random.uniform(0, 2*np.pi, 14)
init_params = init_params.astype(np.float64)
optimized_params, loss = nesterov(train_images_pca, train_labels, init_params)

save_dir = os.makedirs('parameters', exist_ok=True)
save_path = os.path.join(os.getcwd(), save_dir, "optimized_params.npy")
np.save(save_path, optimized_params)
