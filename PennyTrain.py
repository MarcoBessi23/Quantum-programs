import pennylane as qml
from pennylane import numpy as np
from data_preprocessing import PCA_data
import matplotlib.pyplot as plt

dev = qml.device('default.qubit', wires = 4)

def amplitude(f):
    qml.AmplitudeEmbedding(features=f, wires=[1,2,3], normalize=True)

def make_controlled_flip(bitstring):
    def circuit():
        qml.ctrl(lambda: qml.FlipSign(bitstring, wires=[1,2,3]), control=[0])()
    return circuit

controlled_000 = make_controlled_flip([0,0,0])
controlled_001 = make_controlled_flip([0,0,1])
controlled_010 = make_controlled_flip([0,1,0])
controlled_011 = make_controlled_flip([0,1,1])
controlled_100 = make_controlled_flip([1,0,0])
controlled_101 = make_controlled_flip([1,0,1])
controlled_110 = make_controlled_flip([1,1,0])
controlled_111 = make_controlled_flip([1,1,1])

controlled_dict = {
    "000": controlled_000,
    "001": controlled_001,
    "010": controlled_010,
    "011": controlled_011,
    "100": controlled_100,
    "101": controlled_101,
    "110": controlled_110,
    "111": controlled_111,
}

def flexible_oracle(theta):
    keys = list(controlled_dict.keys())
    for i in range(8):
        qml.RX(theta[i], wires=[0])
        controlled_dict[keys[i]]()
        qml.RX(-theta[i], wires=[0])


def diffusion_operator(psi):
    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
    
    qml.CRY(phi=psi[0], wires = [1,2])
    qml.CRY(phi=psi[1], wires = [2,3])
    qml.CRY(phi=psi[2], wires = [3,1])
    
    #qml.Hadamard(wires= [3])
    #qml.Toffoli(wires= [1,2,3])
    #qml.Hadamard(wires= [3])
    qml.CCZ(wires = [1,2,3])

    qml.CRY(phi=psi[3], wires = [3,1])
    qml.CRY(phi=psi[4], wires = [2,3])
    qml.CRY(phi=psi[5], wires = [1,2])

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])


@qml.qnode(dev)
def GQHAN(feature, params):

    qml.AmplitudeEmbedding(features=feature, wires=[1,2,3], normalize=True)
    flexible_oracle(params[:8])
    diffusion_operator(params[8:])
    
    return qml.expval(qml.PauliZ(3))


circuit = GQHAN

def loss_fn(labels, predictions):
    loss = 0
    for y,p in zip(labels, predictions):
        loss += (y-p)**2
    return loss/len(predictions)

def cost(params, X, y):
    pred = [circuit(x, params) for x in X]
    return loss_fn(y, pred)

def accuracy_measure(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

steps = 120
learning_rate = 0.09
batch_size = 30


def circuit_training(X_train, Y_train, X_test, Y_test, num_params):

    params = np.pi * np.random.randn(num_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate, momentum = 0.9)
    loss_history = []

    for it in range(steps):
        print(f'iteration number {it}')
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]
        Y_batch = Y_batch.astype(np.float32)

        params, _, _ = opt.step(cost, params, X_batch, Y_batch)

        if it % 10 == 0:
            cost_new = cost(params, X_train, Y_train)
            loss_history.append(cost_new)
            predictions_train = [np.sign(circuit(x, params)) for x in X_train]
            predictions = [np.sign(circuit(x, params)) for x in X_test]
            accuracy_train = accuracy_measure(Y_train, predictions_train)
            accuracy = accuracy_measure(Y_test, predictions)
            print("iteration: ", it, " cost: ", cost_new, "train accuracy: ", str(accuracy_train), "test accuracy: ", str(accuracy))

    return loss_history, params

#train_images_pca, test_images_pca, train_labels, test_labels = PCA_data()
#train_labels[train_labels == 0] = -1
#test_labels[test_labels == 0] = -1

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd

n_samples = 550 
n_test_samples = 50
n_features = 8  
dataset = 'Fashion'


if dataset == 'Fashion':
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=True)

    pca = PCA(n_components=n_features)
    scaler = StandardScaler()


    y_0 = fashion_mnist.target[(fashion_mnist.target == '0')].sample(n=n_samples, random_state=1)
    y_1 = fashion_mnist.target[(fashion_mnist.target == '1')].sample(n=n_samples, random_state=1)

    X_0 = fashion_mnist.data.iloc[y_0.index]
    X_1 = fashion_mnist.data.iloc[y_1.index]

    y_0 = y_0.to_numpy(dtype=np.int_)
    y_0 = y_0 - 1
    y_1 = y_1.to_numpy(dtype=np.int_)

    X_train = np.concatenate((X_0[:n_samples-n_test_samples], X_1[:n_samples-n_test_samples]))
    y_train = np.concatenate((y_0[:n_samples-n_test_samples], y_1[:n_samples-n_test_samples]))
    X_train, y_train = shuffle(X_train, y_train, random_state=1)

    X_test = np.concatenate((X_0[-n_test_samples:], X_1[-n_test_samples:]))
    y_test = np.concatenate((y_0[-n_test_samples:], y_1[-n_test_samples:]))
    X_test, y_test = shuffle(X_test, y_test, random_state=1)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
elif dataset == 'MNIST':
    mnist = fetch_openml('mnist_784')

    pca = PCA(n_components=n_features)
    scaler = StandardScaler()

    y_0 = mnist.target[(mnist.target == '0')].sample(n=n_samples, random_state=1)
    y_1 = mnist.target[(mnist.target == '1')].sample(n=n_samples, random_state=1)

    X_0 = mnist.data.iloc[y_0]
    X_1 = mnist.data.iloc[y_1]

    y_0 = y_0.to_numpy(dtype=np.int_)
    y_0 = y_0-1
    y_1 = y_1.to_numpy(dtype=np.int_)

    X_train = np.concatenate((X_0[:n_samples-n_test_samples],X_1[:n_samples-n_test_samples]))
    y_train = np.concatenate((y_0[:n_samples-n_test_samples],y_1[:n_samples-n_test_samples]))
    X_train, y_train = shuffle(X_train, y_train, random_state=1)

    X_test = np.concatenate((X_0[-n_test_samples:],X_1[-n_test_samples:]))
    y_test = np.concatenate((y_0[-n_test_samples:],y_1[-n_test_samples:]))
    X_test, y_test = shuffle(X_test, y_test, random_state=1)

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)


loss_history, params = circuit_training(X_train, y_train, X_test, y_test, num_params=14)














# Proof that the matrix corresponds to unitary matrix 16x16 with [ID, lambda(b)] 
#
#@qml.qnode(dev)
#def circuit():
#    controlled_000()
#    return qml.state()
#
#unitary = qml.matrix(circuit)()
#print(unitary)
#def print_unitary(U):
#    np.set_printoptions(precision=1, suppress=True)
#    print(np.real(U))
#
#print_unitary(unitary)

#Proof that circuit corresponds to diffusion operator of GQHAN paper
#@qml.qnode(dev)
#def circuit(psi):
#    diffusion_operator(psi)
#    return qml.state()
#psi_test = np.random.uniform(0, 2*np.pi, 6)
#qml.draw_mpl(circuit)(psi_test)
#plt.show()
#