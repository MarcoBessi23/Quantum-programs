import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi# Numero di qubit
import matplotlib.pyplot as plt


n = 3
qc = QuantumCircuit(n)

# Step 1: Inizializzo con Hadamard tutti i qubit
qc.h([0, 1, 2])

def oracle(qc):
    qc.x(1)       #porta 101 in 111
    qc.h(2)
    qc.ccx(0, 1, 2)  # Toffoli su |101> (trasformato in |111>)
    qc.h(2)
    qc.x(1)       #riporta qubit in 0
    

# Operatore di diffusione (inversione rispetto alla media)
def diffuser(qc):
    qc.h([0, 1, 2])  # Hadamard su tutti i qubit
    qc.x([0, 1, 2])  # NOT su tutti i qubit
    qc.h(2)          # Hadamard sull'ultimo qubit
    qc.ccx(0, 1, 2)  # Toffoli
    qc.h(2)          # Hadamard per chiudere l'operazione
    qc.x([0, 1, 2])  # NOT su tutti i qubit
    qc.h([0, 1, 2])  # Hadamard su tutti i qubit


#numero iterazioni = (pi/4)*sqrt(N/m) vale solo quando N è molto maggiore del numero delle soluzioni m, 
# quindi quando theta è molto piccolo
# Iterazione dell'algoritmo di Grover (2 iterazioni)
oracle(qc)      # Prima iterazione dell'oracolo
diffuser(qc)    # Prima iterazione del diffusore


# Misurazione
qc.measure_all()
#qc.draw(output="mpl", style="iqp")


# Transpile for simulator
simulator = AerSimulator()
circ = transpile(qc, simulator)

## Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)

plot_histogram(counts)
plt.show()





## Construct an ideal simulator with SamplerV2
sampler = SamplerV2()
job = sampler.run([circ], shots=1024)

# Perform an ideal simulation
result_ideal = job.result()
counts_ideal = result_ideal[0].data.meas.get_counts()
print('Counts(ideal):', counts_ideal)
