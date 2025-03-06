import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
##BUILD THE NEURAL NETWORK
def amplitude_encoding(input_vector:np.array):
    '''
    Amplitude encoding for a vecto of size 8, it takes |000> and 
    the input vector and brings it in the desired encoded state.
    '''
    
    norm = np.linalg.norm(input_vector)
    quantum_state = input_vector/norm
    
    def calculate_theta(values):
        """Procedure that returns the angle for RY"""
        norm_factor = np.linalg.norm(values)
        return 2 * np.arctan(np.sqrt(sum(values[len(values)//2:]**2) / sum(values[:len(values)//2]**2)))


    ###HERE I WANT TO APPLY THE ROTATION NECESSARY TO GET FROM |000> TO THE ENCODED STATE I USE THE TREE METHOD
    theta_1 = calculate_theta(quantum_state)      
    theta_2 = calculate_theta(quantum_state[:4])  
    theta_3 = calculate_theta(quantum_state[4:])  
    theta_4 = calculate_theta(quantum_state[:2])  
    theta_5 = calculate_theta(quantum_state[2:4])
    theta_6 = calculate_theta(quantum_state[4:6])
    theta_7 = calculate_theta(quantum_state[6:])

    qc = QuantumCircuit(4, 3)
    qc.ry(theta_1, 1) 
    qc.cx(1, 2)
    qc.ry(theta_2, 2)
    qc.ry(theta_3, 2)
    qc.cx(2, 3)
    qc.ry(theta_4, 3)
    qc.ry(theta_5, 3)
    qc.ry(theta_6, 3)
    qc.ry(theta_7, 3)

    return qc


def oracle_gate_000():
    """Oracolo che inverte la fase di |000>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.x(0)
    oracle.x(1)
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)
    oracle.x(1)
    oracle.x(0)

    return oracle.to_gate()


def oracle_gate_001():
    """Oracolo che inverte la fase di |001>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.x(0)
    oracle.x(1)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(1)
    oracle.x(0)

    return oracle.to_gate()


def oracle_gate_010():
    """Oracolo che inverte la fase di |010>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.x(0)
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)
    oracle.x(0)

    return oracle.to_gate()

def oracle_gate_011():
    """Oracolo che inverte la fase di |011>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.x(0)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(0)

    return oracle.to_gate()


def oracle_gate_100():
    """Oracolo che inverte la fase di |100>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.x(1)
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)
    oracle.x(1)
    
    return oracle.to_gate()

def oracle_gate_101():
    """Oracolo che inverte la fase di |101>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.x(1)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(1)
    
    return oracle.to_gate()

def oracle_gate_110():
    """Oracolo che inverte la fase di |110>"""
    oracle = QuantumCircuit(3, name="Oracle")
    #oracle.x(0)
    #oracle.x(1)
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)
    #oracle.x(1)
    #oracle.x(0)

    return oracle.to_gate()

def oracle_gate_111():
    """Oracolo che inverte la fase di |111>"""
    oracle = QuantumCircuit(3, name="Oracle")
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    
    return oracle.to_gate()

def controlled_oracle(qbit_str, oracle_gate):
    """Oracolo controllato per |100>"""
    control = QuantumRegister(1, "control")  # Qubit di controllo
    target = QuantumRegister(3, "target")    # Qubits per l'oracolo
    qc = QuantumCircuit(control, target)

    oracle = oracle_gate()
    controlled_oracle = oracle.control(1)

    qc.append(controlled_oracle, [control[0], target[0], target[1], target[2]])
    return qc.to_gate(label=f"ControlledOracle{qbit_str}]")


controlled_gate_000 = controlled_oracle('000', oracle_gate_000)
controlled_gate_001 = controlled_oracle('001', oracle_gate_001)
controlled_gate_010 = controlled_oracle('010', oracle_gate_010)
controlled_gate_011 = controlled_oracle('011', oracle_gate_011)
controlled_gate_100 = controlled_oracle('100', oracle_gate_100)
controlled_gate_101 = controlled_oracle('101', oracle_gate_101)
controlled_gate_110 = controlled_oracle('110', oracle_gate_110)
controlled_gate_111 = controlled_oracle('111', oracle_gate_111)



#https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/grover.ipynb
def flexible_oracle(qc):

    theta = ParameterVector('theta', length = 8)

    #|000>
    qc.rx(theta[0], 0)
    qc.append(controlled_gate_000, [0,1,2,3])

    #|001>
    qc.rx(theta[1], 0)
    qc.append(controlled_gate_001, [0,1,2,3])

    #|010>
    qc.rx(theta[2], 0)
    qc.append(controlled_gate_010, [0,1,2,3])
    
    #|011>
    qc.rx(theta[3], 0)
    qc.append(controlled_gate_011, [0,1,2,3])

    #|100>
    qc.rx(theta[4], 0)
    qc.append(controlled_gate_100, [0,1,2,3])
    
    #|101>
    qc.rx(theta[5],0)
    qc.append(controlled_gate_101, [0,1,2,3])
    
    #|110>
    qc.rx(theta[6],0)
    qc.append(controlled_gate_110, [0,1,2,3])
    
    #|111>
    qc.rx(theta[7],0)
    qc.append(controlled_gate_111, [0,1,2,3])

    return qc

def adaptive_diffusion(qc):
    ''' 
    psi is the training parameter
    '''

    ### FIRST COSTRUCT THE MULTICONTROL ZGATE TO USE IN THE NETWORK
    #zcir = QuantumCircuit(1)
    #zcir.z(0)
    #zgate = zcir.to_gate(label='z').control(3, ctrl_state= '111')

    psi = ParameterVector('psi', length = 6)
    qc.h([1,2,3])
    qc.cry(psi[0], 1, 2)
    qc.cry(psi[1], 2, 3)
    qc.cry(psi[2], 3, 1)
    ##MULTI CONTROLLED Z GATE
    qc.h(3)
    qc.ccx(1, 2, 3)
    qc.h(3)
    qc.cry(psi[3], 3, 1)
    qc.cry(psi[4], 2, 3)
    qc.cry(psi[5], 1, 2)
    qc.barrier()
    qc.h([1,2,3])

    return qc

def GQHAN(input):
    '''
    Grover inspired Quantum Attention Network from https://arxiv.org/abs/2401.14089
    '''
    
    qc = QuantumCircuit(4)
    initial_state = input/np.linalg.norm(input)
    amplitude = StatePreparation(initial_state) ##Amplitude Encoding
    qc.append(amplitude, [1,2,3])
    qc = flexible_oracle(qc)
    qc = adaptive_diffusion(qc)

    return qc


def ansatz(qc):
    qc = flexible_oracle(qc)
    qc = adaptive_diffusion(qc)
    return qc

#def GQHAN():
#    '''
#    Grover inspired Quantum Attention Network from https://arxiv.org/abs/2401.14089
#    '''
#    
#    qc = QuantumCircuit(4) #il primo qubit serve per altre applicazioni, usare solo 1,2,3
#    x = ParameterVector('x', length = 8)
#    amplitude = StatePreparation(x) ##Amplitude Encoding
#    qc.append(amplitude, [1,2,3])
#    qc = flexible_oracle(qc)
#    qc = adaptive_diffusion(qc)
#
#    return qc



def diffuser(qc):
    qc.h([1, 2, 3])  # Hadamard su tutti i qubit
    qc.x([1, 2, 3])  # NOT su tutti i qubit
    qc.h(3)          # Hadamard sull'ultimo qubit
    qc.ccx(1, 2, 3)  # Toffoli
    qc.h(3)          # Hadamard per chiudere l'operazione
    qc.x([1, 2, 3])  # NOT su tutti i qubit
    qc.h([1, 2, 3])  # Hadamard su tutti i qubit

#qc = GQHAN()
#print(qc.draw('text'))

#qc_test = QuantumCircuit(4)  # Deve avere il numero corretto di qubit
#qc_test.x(0)
#qc_test.h([1,2,3])
#qc_test.append(controlled_gate_111, [0, 1, 2, 3])
#diffuser(qc_test)
#qc_test.measure_all()
#
#from qiskit import QuantumCircuit, transpile
#from qiskit_aer import AerSimulator
## Transpile for simulator
#simulator = AerSimulator()
#circ = transpile(qc_test, simulator)
#
### Run and get counts
#result = simulator.run(circ).result()
#counts = result.get_counts(circ)
#
#import matplotlib.pyplot as plt
#from qiskit.visualization import plot_histogram
#plot_histogram(counts)
#plt.show()