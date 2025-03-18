import os
import numpy as np

from data_preprocessing import PCA_data
from quantum_NeuralNet import prepare_angles, gqhan, small_gqhan
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient, LinCombEstimatorGradient
from qiskit_machine_learning.optimizers import COBYLA, SPSA, QNSPSA, SciPyOptimizer, ADAM
from qiskit.quantum_info import SparsePauliOp, Pauli
import matplotlib.pyplot as plt



init_params = np.random.uniform(0, 2*np.pi, 8) #14)
init_params = init_params.astype(np.float64)
train_images_pca, test_images_pca, train_labels, test_labels = PCA_data()
train_labels[train_labels == 0] = -1
test_labels[test_labels == 0] = -1
train_angles = np.array([prepare_angles(img) for img in train_images_pca])
print(np.shape(train_angles))
test_angles = np.array([prepare_angles(img) for img in test_images_pca])
print('FINE PREPROCESSING ANGOLI')

observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])
#observable = Pauli('ZIII')
qc = gqhan()
#qc = small_gqhan()
estimator = Estimator()

estimator_qnn = EstimatorQNN(circuit = qc.decompose(),
                             estimator = estimator,
                             gradient = ParamShiftEstimatorGradient(estimator),
                             #gradient = LinCombEstimatorGradient(estimator),
                             #gradient = SPSAEstimatorGradient(estimator),
                             
                             observables = observable,
                             input_params = qc.parameters[:7],
                             weight_params= qc.parameters[7:]
                             )


class Classifier():

    def __init__(self, estimator, initial_parameters, train_data, train_labels, epochs, batch_size):
        self.derivative_mse = []
        self.estimator = estimator
        self.X_train = train_data
        self.y_train = train_labels
        self.epochs = epochs
        self.loss_values = []
        self.parameters = initial_parameters
        self.batch_size = batch_size 

    def loss_fn(self, x, y, w):

        y_pred = self.estimator.forward(x, w)
        y_pred = y_pred.reshape(np.shape(y))
        l = np.mean((y_pred-y)**2)
        self.derivative_mse.append((2 / len(y)) * (y_pred - y))
        return l     

    def compute_gradient(self, x, w):

        _, g = self.estimator.backward(x, w)
        gradient = g.squeeze(axis=1) #NEEDED BECAUSE BACKWARD GRADIENT HAS SHAPE (BATCH, 1, NUM_PARAMETERS) (?)
        gradient *= np.array(self.derivative_mse).transpose() #NEEDED BECAUSE derivative_mse HAS SHAPE (1,30)
        self.derivative_mse = []
        
        return np.sum(gradient)

    def train_model_nesterov(self, momentum= 0.9, learning_rate = 0.09):
        velocity = np.zeros_like(self.parameters)
        w = np.copy(self.parameters)

        for epoch in range(self.epochs):
            print(f'EPOCH NUMBER {epoch}')
            indices = np.random.permutation(len(self.X_train))
            train_data, train_labels = self.X_train[indices], self.y_train[indices]
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size]
                batch_labels = train_labels[i : i + self.batch_size]            
                lookahead_params = w - momentum * velocity
                loss = self.loss_fn(batch, batch_labels, lookahead_params)
                gradient = self.compute_gradient(batch,lookahead_params)
                velocity = momentum * velocity + learning_rate * gradient
                w -= velocity
                self.loss_values.append(loss)
                print(f'loss at iteration {len(self.loss_values)}: {loss}')

        self.parameters = np.copy(w)

    def train_COBYLA(self):
        
        def objective_fun(self):
            
            pass

    def plot_training_loss(self, path):
        
        plt.figure(figsize=(8, 5))
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(self.loss_values, marker='o', linestyle='-')
        plt.grid()
        plt.savefig(path)

    def score(self, x_test, y_test):
        y_pred = self.estimator.forward(x_test, self.parameters)
        y_pred = y_pred.reshape(np.shape(y_test))
        prediction = np.sign(y_pred)
        return np.sum(prediction == y_test)/len(y_test)


if not os.path.exists("results"):
            os.makedirs("results")
path_train_loss = "results/training_results_nesterov_small_8_params.png"


gqhan_classifier = Classifier(estimator_qnn, init_params, train_angles, train_labels, epochs = 3, batch_size= 30)
gqhan_classifier.train_model_nesterov()
gqhan_classifier.plot_training_loss(path_train_loss)
accuracy = gqhan_classifier.score(test_angles, test_labels)
print(accuracy)
