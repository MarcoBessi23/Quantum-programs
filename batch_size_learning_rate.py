import pennylane as qml
from pennylane import numpy as np
from Pennylane_QNN import flexible_oracle, diffusion_operator
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description="Quantum circuit training")
parser.add_argument('--batch_size', type=int, default=30, help="Size of the training batch")
parser.add_argument('--learning_rate', type=float, default=0.09, help="Learning rate for the optimizer")
args = parser.parse_args()

num_layers = 1
steps = 240
batch_size = args.batch_size
learning_rate = args.learning_rate
num_params = 14
n_samples = 550 
n_test_samples = 50
n_features = 8  
dataset = 'Fashion'



dev = qml.device('default.qubit', wires = 4)


@qml.qnode(dev)
def circuit(feature, params):

    qml.AmplitudeEmbedding(features=feature, wires=[1,2,3], normalize=True)
    for layer in range(num_layers):
        flexible_oracle(params[:8])
        diffusion_operator(params[8:])
    
    return qml.expval(qml.PauliZ(3))





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




def circuit_training(X_train, Y_train, X_test, Y_test):
    
    np.random.seed(42) 
    #params = np.pi * np.random.randn(num_params, requires_grad=True)
    params = 0.01 * np.random.randn(num_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []
    acc_test_history = []
    acc_train_history = []
    training_step = []

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
            print('previsioni del modello')
            print(predictions)
            print('y_test')
            print(Y_test)
            accuracy_train = accuracy_measure(Y_train, predictions_train)
            acc_train_history.append(accuracy_train)
            accuracy_test = accuracy_measure(Y_test, predictions)
            acc_test_history.append(accuracy_test)
            training_step.append(it)
            print("iteration: ", it, " cost: ", cost_new,"train accuracy: ", str(accuracy_train),  "test accuracy: ", str(accuracy_test))
            
    plt.plot(training_step, acc_train_history, color='blue', label='Train Accuracy')
    plt.plot(training_step, acc_test_history, color='red', label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.savefig(f'results/Accuracy_batch{batch_size}_lr{learning_rate}.png')
    plt.close()

    plt.plot(training_step, loss_history, color='green', label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Train Loss')
    plt.savefig(f'results/training_loss_batch{batch_size}_lr{learning_rate}.png')
    plt.close()

    return loss_history, params


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
    print('vettore di y0')
    print(y_0)
    y_1 = y_1.to_numpy(dtype=np.int_)
    print('vettore di y1')
    print(y_1)

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




if __name__ == "__main__":


    loss_history, params = circuit_training(X_train, y_train, X_test, y_test)
    pred_final = [np.sign(circuit(x, params)) for x in X_test]
    print('Test Accuracy finale')
    print(accuracy_measure(labels = y_test, predictions = pred_final))

