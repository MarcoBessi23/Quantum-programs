import os
import numpy as np
from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.primitives import Estimator, StatevectorEstimator
from data_preprocessing import PCA_data
from quantum_NeuralNet import GQHAN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

train_images_pca, test_images_pca, train_labels, test_labels = PCA_data()
train_labels[train_labels == 0] = -1
test_labels[test_labels == 0] = -1


#def accuracy(params, input_data, labels):
#    y_pred = []
#    for i, x in enumerate(input_data):
#        
#        qc = GQHAN(x)
#        z_op = Pauli("ZIII") ###see https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli
#        estimator = Estimator()
#        job = estimator.run([qc], [z_op], [params])
#        expectation_value = job.result().values[0]
#        y_pred.append(np.sign(expectation_value))
#    
#    acc = np.sum(labels == y_pred)/len(labels)
#    return acc
#
#
#
#def cost_function(params, batch, labels):
#    '''
#    Calculate loss function as shown in https://arxiv.org/abs/2401.14089    
#    '''
#    dL_df = []
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
#        #total_cost += (y-np.sign(expectation_value))**2
#        total_cost += (y-expectation_value)**2
#        #print('EXPECTATION VALUE')
#        #print(np.sign(expectation_value))
#        #first_derivative = y-np.sign(expectation_value)
#        first_derivative = y-expectation_value
#        dL_df.append(first_derivative)
#    return total_cost / len(batch), -2 * np.array(dL_df) 
#
#def compute_gradient(params:np.ndarray, batch, first_derivative):
#    '''
#    compute gradient of the quantum part of the loss using parameter shift
#    '''
#
#    estimator = Estimator()
#    z_op = Pauli("ZIII")
#    grad = np.zeros_like(params)
#    print('Gradient calculation')
#    for i, x in enumerate(batch):
#        qc = GQHAN(x)
#        gradient = SPSAEstimatorGradient(estimator, epsilon=0.05)
#        pse_grad_result = gradient.run(qc, z_op, [params.tolist()]).result().gradients
#        g = np.array(pse_grad_result)[0]
#        g *= first_derivative[i]
#        grad += g
#        #print('GRADIENT VALUE')
#        #print(g)
#        
#    return grad/len(batch)

#from qiskit_algorithms.gradients.utils import DerivativeType
#
#def compute_gradient(params:np.ndarray, batch, first_derivative):
#    '''
#    compute gradient of the quantum part of the loss using parameter shift
#    '''
#
#    
#    
#    z_op = Pauli("ZIII")
#    grad = np.zeros_like(params)
#    print('Gradient calculation')
#    for i, x in enumerate(batch):
#        qc = GQHAN(x)
#        estimator = StatevectorEstimator()
#        gradient = ReverseQGT(estimator, derivative_type=DerivativeType.REAL)
#        pse_grad_result = gradient.run(qc,[params.tolist()]).result().gradients
#        g = np.array(pse_grad_result)[0]
#        g *= first_derivative[i]
#        grad += g
#        #print('GRADIENT VALUE')
#        #print(g)
#        
#    return grad/len(batch)
#
#
##init_params = np.random.uniform(0, 2*np.pi, 14)
##init_params = init_params.astype(np.float64)
##batch = train_images_pca[0 : 30]
##batch_labels = train_labels[0 : 30]
#first_derivative = np.ones((30,))
#compute_gradient(init_params, batch, first_derivative)

#def nesterov(train_data, train_labels, init_params, learning_rate=0.001, momentum=0.9, epochs=4, batch_size=30):
#
#    params = init_params
#    velocity = np.zeros_like(params)
#
#    for epoch in range(epochs):
#        print(f'EPOCH NUMBER {epoch}')
#        indices = np.random.permutation(len(train_data))
#        train_data, train_labels = train_data[indices], train_labels[indices]
#        loss_values = []
#
#        for i in range(0, len(train_data), batch_size):
#            print(i)
#            batch = train_data[i : i + batch_size]
#            batch_labels = train_labels[i : i + batch_size]            
#            lookahead_params = params - momentum * velocity
#
#            loss, dL_df = cost_function(lookahead_params, batch, batch_labels)
#            loss_gradient = compute_gradient(lookahead_params, batch, dL_df)
#            velocity = momentum * velocity + learning_rate * loss_gradient
#            params -= velocity
#            
#            if i%60 == 0:
#                print(f'loss at iteration {i} is {loss:.4f}')
#                loss_values.append(loss)
#        print('Evaluating test accuracy')
#        acc = accuracy(params, test_images_pca, test_labels)
#        print(f'test accuracy at epoch {epoch}: {acc:.2f}')
#
#    return params, loss_values, acc

#np.random.seed(42)
#init_params = np.random.uniform(0, 2*np.pi, 14)
#init_params = init_params.astype(np.float64)
#optimized_params, loss, acc = nesterov(train_images_pca, train_labels, init_params)
#save_dir = os.makedirs('parameters', exist_ok=True)
#save_path = os.path.join(os.getcwd(), save_dir, "optimized_params.npy")
#np.save(save_path, optimized_params)


from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit.library import StatePreparation
from quantum_NeuralNet import ansatz

qc = QuantumCircuit(4)
amplitude = RawFeatureVector(8)
qc.append(amplitude, [1, 2, 3])
qc = ansatz(qc)
print(qc.parameters)
print(qc.draw('text'))
x = train_images_pca[0]
init_params = np.random.uniform(0, 2*np.pi, 14)
init_params = init_params.astype(np.float64)
z_op = Pauli("ZIII")
estimator_qnn = EstimatorQNN(circuit= qc,
                             #estimator = estimator,
                             input_params = qc.parameters[14:],
                             weight_params= qc.parameters[0:14])



x = train_images_pca[0:30]
x = x/np.linalg.norm(x)
p1 = estimator_qnn.forward(input_data=x, weights=init_params)
estimator_qnn.backward(input_data = x, weights = init_params)