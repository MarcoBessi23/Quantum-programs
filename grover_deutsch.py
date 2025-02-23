import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt


n = 4
qc = QuantumCircuit(n)

#010->1
#101->1
#011->1
#110->1 
#Quindi dalle slides per fare la funzione devo costruire un circuito per cui se per esempio entra 110y 
# allora deve uscire 110(NOTy)


def Uf(qc):
    #101
    qc.x(1) #qui ho 111
    qc.cx(0,3) #-y
    qc.cx(1,3) #y
    qc.cx(2,3) #-y 
    qc.x(1) #riporto tutto nello stato di partenza

    #011
    qc.x(0)
    qc.cx(0,3) #-y
    qc.cx(1,3) #y
    qc.cx(2,3) #-y 
    qc.x(0) #riporto tutto nello stato di partenza



zero_f = QuantumCircuit(1)

one_f = QuantumCircuit(1)
one_f.x(0)

# Step 1: Inizializzo con Hadamard i primi tre qubit
qc.h([0, 1, 2])
qc.x(4) #porto il primo qubit |0> in |1>
qc.h(4) #applico Hadamard a |1> 
Uf(qc)
qc.h([0, 1, 2, 3])

def oracle(qc):
    pass

def diffuser(qc):
    pass