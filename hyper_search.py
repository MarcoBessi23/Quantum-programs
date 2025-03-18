import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from data_preprocessing import PCA_data
from quantum_NeuralNet import  prepare_angles, gqhan
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split


train_images_pca, test_images_pca, train_labels, test_val_labels = PCA_data()
train_labels[train_labels == 0] = -1
test_val_labels[test_val_labels == 0] = -1


print(test_val_labels)
print(len(test_val_labels[test_val_labels == -1]))

train_angles = np.array([prepare_angles(img) for img in train_images_pca])
print(np.shape(train_angles))
test_val_angles = np.array([prepare_angles(img) for img in test_images_pca])
print('END ANGLES PREPROCESSING ')
val_angles, test_angles, val_labels, test_labels = train_test_split(
    test_val_angles,
    test_val_labels, 
    test_size = 100, 
    stratify = test_val_labels, 
    random_state = 42
)

class Random_Search():
    '''
    Class to implement Random Search to fine tune initial parameters value
    '''
    def __init__(self, num_samples):
            
        self.num_samples = num_samples
        self.configuration_space = np.random.uniform(-np.pi, np.pi, (num_samples, 14))
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])
        self.estimator = StatevectorEstimator()
        self.qc = QuantumCircuit(4)
        self.qc = gqhan()
        self.training_iteration = 0
        self.scores = []
        self.neural_network  = EstimatorQNN(circuit = self.qc.decompose(),
                                     estimator = self.estimator,
                                     observables = self.observable,
                                     input_params = self.qc.parameters[:7],
                                     weight_params = self.qc.parameters[7:]
                                     )
        self.classifier = NeuralNetworkClassifier(neural_network = self.neural_network,
                                         optimizer = COBYLA(maxiter=20),
                                         callback = self.callback_graph
                                         )
    
    def callback_graph(self, weights, obj_func_eval):
        self.training_iteration += 1
        print(f"Iterazione {self.training_iteration}: {obj_func_eval}")

    def calculate_score_function(self, hyper):
        self.classifier.initial_point = hyper
        self.classifier.fit(train_angles, train_labels)
        score = self.classifier.score(val_angles, val_labels)
        self.scores.append(score)


    def search(self):

        for i in range(self.num_samples):
            print(f'Sample number {i} of {self.num_samples-1}')
            self.calculate_score_function(self.configuration_space[i])
            self.training_iteration = 0
        index_best = np.argmax(self.scores)
        best_hyper = self.configuration_space[index_best]
        scores = np.array(self.scores)
        np.save('scores_list.npy', scores)
        
        return best_hyper

rs = Random_Search(num_samples = 15)
best_hyper = rs.search()
print(f'BEST STARTING VALUE IS : {best_hyper}')
np.save("best_initial_params.npy", best_hyper)
