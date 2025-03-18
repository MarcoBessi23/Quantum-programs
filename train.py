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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



train_images_pca, test_images_pca, train_labels, test_val_labels = PCA_data()
train_labels[train_labels == 0] = -1
test_val_labels[test_val_labels == 0] = -1
train_angles = np.array([prepare_angles(img) for img in train_images_pca])
test_val_angles = np.array([prepare_angles(img) for img in test_images_pca])
print('END ANGLES PREPROCESSING ')
val_angles, test_angles, val_labels, test_labels = train_test_split(
    test_val_angles,
    test_val_labels,
    test_size = 100,
    stratify = test_val_labels,
    random_state = 42
)

observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])
qc = QuantumCircuit(4)
qc = gqhan()
estimator = StatevectorEstimator()
init_params = np.load('best_initial_params.npy')


class gqhan_classifier():
    
    def __init__(self, estimator, qc, observable, train_angles, test_angles, train_labels, test_labels, init_params):
        self.estimator_qnn = EstimatorQNN(circuit= qc.decompose(),
                                          estimator = estimator,
                                          observables= observable,
                                          input_params = qc.parameters[0:7],
                                          weight_params= qc.parameters[7:]
                                          )
        self.X_train = train_angles
        self.X_test  = test_angles
        self.y_train = train_labels
        self.y_test = test_labels
        self.initial_parameters = init_params
        self.objective_func_vals = []
        self.classifier_qnn = NeuralNetworkClassifier(neural_network=self.estimator_qnn,
                                                      optimizer = COBYLA(maxiter= 50),
                                                      callback= self.callback
                                                    )

    def callback(self, weights, obj_func_eval):
        iteration = len(self.objective_func_vals)
        print(f"Iterazione {iteration}: {obj_func_eval}")
        self.objective_func_vals.append(obj_func_eval)

    def train(self):
        self.classifier_qnn.fit(self.X_train, self.y_train)
        self.classifier_qnn.save('GQHAN_COBYLA.model')
        np.save('final_parameters.npy', qc.parameters[7:])

    def plot_results(self, path):
        plt.figure(figsize=(8, 5))
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals, marker='o', linestyle='-')
        plt.grid()
        plt.savefig(path)
        plt.close()

    def get_accuracy(self):
        return self.classifier_qnn.score(self.X_test, self.y_test)
    



classifier = gqhan_classifier(estimator = estimator, 
                              qc = qc, 
                              observable = observable, 
                              train_angles = train_angles, 
                              test_angles = test_angles, 
                              train_labels = train_labels, 
                              test_labels = test_labels, 
                              init_params = init_params
                              )
classifier.train()
classifier.plot_results('results/COBYLA_training_plot.png')
classifier.plot_results('results/COBYLA_training_plot_cross_entropy_loss.png')
acc = classifier.get_accuracy()
print('ACCURACY:')
print(acc)