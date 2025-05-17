import pennylane as qml
from pennylane import numpy as np


def make_controlled_flip(bitstring):
    def controlled_circuit():
        qml.ctrl(lambda: qml.FlipSign(bitstring, wires=[1,2,3]), control=[0])()
    return controlled_circuit


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

def compressed_flexible_oracle(theta):
    keys = list(controlled_dict.keys())
    for i in range(8):
        if i%2==0:
            qml.RX(theta[i//2], wires=[0])
            controlled_dict[keys[i]]()
            qml.RX(-theta[i//2], wires=[0])
        else :
            controlled_dict[keys[i]]()

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
    #qml.CCZ(wires = [1,2,3])

    qml.CRY(phi=psi[3], wires = [3,1])
    qml.CRY(phi=psi[4], wires = [2,3])
    qml.CRY(phi=psi[5], wires = [1,2])

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])

def diffusion_less_parameters(psi):
    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
    
    qml.CRY(phi=psi, wires = [1,2])
    qml.CRY(phi=psi, wires = [2,3])
    qml.CRY(phi=psi, wires = [3,1])
    
    qml.Hadamard(wires= [3])
    qml.Toffoli(wires= [1,2,3])
    qml.Hadamard(wires= [3])
    #qml.CCZ(wires = [1,2,3])

    qml.CRY(phi=psi, wires = [3,1])
    qml.CRY(phi=psi, wires = [2,3])
    qml.CRY(phi=psi, wires = [1,2])

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])



def diffusion_RX(psi):
    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
    
    qml.CRX(phi=psi[0], wires = [1,2])
    qml.CRX(phi=psi[1], wires = [2,3])
    qml.CRX(phi=psi[2], wires = [3,1])
    
    qml.Hadamard(wires= [3])
    qml.Toffoli(wires= [1,2,3])
    qml.Hadamard(wires= [3])
    #qml.CCZ(wires = [1,2,3])

    qml.CRX(phi=psi[3], wires = [3,1])
    qml.CRX(phi=psi[4], wires = [2,3])
    qml.CRX(phi=psi[5], wires = [1,2])

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])


def diffusion_block(psi):

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
    
    qml.CRY(phi=psi[0], wires = [1,2])
    qml.CRY(phi=psi[1], wires = [2,3])
    qml.CRY(phi=psi[2], wires = [3,1])
    
    qml.RZ(psi[3], wires=1)
    qml.RY(psi[4], wires=2)
    qml.RX(psi[5], wires=3)

    qml.Hadamard(wires= [3])
    qml.Toffoli(wires= [1,2,3])
    qml.Hadamard(wires= [3])
    #qml.CCZ(wires = [1,2,3])

    qml.RZ(psi[6], wires=1)
    qml.RY(psi[7], wires=2)
    qml.RX(psi[8], wires=3)

    qml.CRY(phi=psi[9], wires = [3,1])
    qml.CRY(phi=psi[10], wires = [2,3])
    qml.CRY(phi=psi[11], wires = [1,2])

    qml.Hadamard(wires = [1])
    qml.Hadamard(wires = [2])
    qml.Hadamard(wires = [3])
