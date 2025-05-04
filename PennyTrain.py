import pennylane as qml
from pennylane import numpy as np
from data_preprocessing import PCA_data

dev = qml.device('default.qubit', wires = 4)
#@qml.qnode(dev)
def Amplitude(f):
    qml.AmplitudeEmbedding(features=f, wires=[1,2,3], normalize=True)
    return qml.state()

#https://pennylane.ai/qml/demos/tutorial_grovers_algorithm

#@qml.qnode(dev)
#def circuit_000():
#    # Flipping the marked state
#    qml.FlipSign([0, 0, 0], wires=[1,2,3])
#    return qml.state()
#
#@qml.qnode(dev)
#def controlled_000():
#    qml.ctrl(circuit_000, control = [0], control_values = 1)
#    return qml.state()

#drawer = qml.draw(controlled_000, show_all_wires = True)
#print(drawer())


def make_controlled_flip(bitstring):
    @qml.qnode(dev)
    def circuit():
        qml.ctrl(lambda: qml.FlipSign(bitstring, wires=[1,2,3]), control=[0])()
        return qml.state()
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

@qml.qnode(dev)
def flexible_oracle(theta):
    keys = list(controlled_dict.keys())
    for i in range(8):
        qml.RX(theta[i], wires=0)
        controlled_dict[keys[i]]()
    return qml.state()


@qml.qnode(dev)
def diffusion_operator(psi):
    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
    
    qml.CRY(phi=psi[0], wires = [1,2])
    qml.CRY(phi=psi[1], wires = [2,3])
    qml.CRY(phi=psi[2], wires = [3,1])
    
    qml.Hadamard(wires= [3])
    qml.Toffoli(wires= [1,2,3])
    qml.Hadamard(wires= [3])

    qml.CRY(phi=psi[4], wires = [3,1])
    qml.CRY(phi=psi[5], wires = [2,3])
    qml.CRY(phi=psi[6], wires = [1,2])

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
    
    return qml.state()

#@qml.node(dev)
#def flexible_oracle(theta):
#    qml.RX(phi = theta[0], wires= [0])
#    controlled_000()
#    qml.RX(phi = theta[1], wires= [0])
#    controlled_001()
#    qml.RX(phi = theta[2], wires= [0])
#    controlled_010()
#    qml.RX(phi = theta[3], wires= [0])
#    controlled_011()
#    qml.RX(phi = theta[4], wires= [0])
#    controlled_100()
#    qml.RX(phi = theta[5], wires= [0])
#    controlled_101()
#    qml.RX(phi = theta[6], wires= [0])
#    controlled_110()
#    qml.RX(phi = theta[7], wires= [0])
#    controlled_111()
#
#    return qml.state()


import matplotlib.pyplot as plt

psi = np.random.uniform(0, 2*np.pi, 7)

drawer = qml.draw_mpl(diffusion_operator)
drawer(psi)
plt.show()