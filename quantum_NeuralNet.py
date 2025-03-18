import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate
import matplotlib.pyplot as plt


##BUILD THE NEURAL NETWORK

def amplitude_encoding():
    
    angles = ParameterVector('alpha', length=7)
    qc = QuantumCircuit(4)
    
    qc.ry(angles[0],3)
    qc.barrier()

    qc.cry(angles[1],3,2)

    qc.x(3)
    qc.cry(angles[2], 3, 2)
    qc.x(3)
    qc.barrier()

    ccry = RYGate(angles[6]).control(2)
    qc.append(ccry,[3,2,1])

    ccry = RYGate(angles[5]).control(2)
    qc.x(2)
    qc.append(ccry,[3,2,1])
    qc.x(2)

    qc.x(3)
    ccry = RYGate(angles[4]).control(2)
    qc.append(ccry,[3,2,1])
    ccry = RYGate(angles[3]).control(2)
    qc.x(2)
    qc.append(ccry,[3,2,1])
    qc.x(2)
    qc.x(3)
    
    return qc





def oracle_gate_000():
    """Oracolo che inverte la fase di q1q2q3 =|000>"""
    oracle = QuantumCircuit(3, name="Oracle000")
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
    """Oracolo che inverte la fase di q1q2q3 =|001>"""
    oracle = QuantumCircuit(3, name="Oracle001")
    oracle.x(0)
    oracle.x(1)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(1)
    oracle.x(0)

    return oracle.to_gate()


def oracle_gate_010():
    """Oracolo che inverte la fase di q1q2q3 =|010>"""
    oracle = QuantumCircuit(3, name="Oracle010")
    oracle.x(0)
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)
    oracle.x(0)

    return oracle.to_gate()

def oracle_gate_011():
    """Oracolo che inverte la fase di q1q2q3 =|011>"""
    oracle = QuantumCircuit(3, name="Oracle011")
    oracle.x(0)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(0)

    return oracle.to_gate()


def oracle_gate_100():
    """Oracolo che inverte la fase di q1q2q3 =|100>"""
    oracle = QuantumCircuit(3, name="Oracle100")
    oracle.x(1)
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)
    oracle.x(1)
    
    return oracle.to_gate()

def oracle_gate_101():
    """Oracolo che inverte la fase di q1q2q3 =|101>"""
    oracle = QuantumCircuit(3, name="Oracle101")
    oracle.x(1)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(1)
    
    return oracle.to_gate()

def oracle_gate_110():
    """Oracolo che inverte la fase di q1q2q3 = |110>"""
    oracle = QuantumCircuit(3, name="Oracle110")
    oracle.x(2)
    oracle.h(2)
    oracle.ccx(0, 1, 2)
    oracle.h(2)
    oracle.x(2)

    return oracle.to_gate()

def oracle_gate_111():
    """Oracolo che inverte la fase di q1q2q3 =|111>"""
    oracle = QuantumCircuit(3, name="Oracle111")
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
    theta = ParameterVector('beta', length = 8)

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
    psi = ParameterVector('gamma', length = 6)
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



def prepare_angles(input_data):
    ##Function needed to prepare the angles
    #[000, 001, 010, 011, 100, 101, 110, 111]
    #Normalize data
    theta = []
    input_data = input_data/np.linalg.norm(input_data)
    
    prob_0_q2 = np.linalg.norm(input_data[0:4])
    theta_0 = 2*np.arccos(prob_0_q2)
    theta.append(theta_0)
    
    prob_10_q2q1 = np.linalg.norm(input_data[4:6])/np.linalg.norm(input_data[4:])
    theta_1 = 2*np.arccos(prob_10_q2q1)
    theta.append(theta_1)
    
    prob_00_q2q1 = np.linalg.norm(input_data[0:2])/np.linalg.norm(input_data[:4])
    theta_2 = 2*np.arccos(prob_00_q2q1)
    theta.append(theta_2)

    prob_000_q2q1q0 = np.linalg.norm(input_data[0])/np.linalg.norm(input_data[0:2])
    theta_3 = 2*np.arccos(prob_000_q2q1q0)
    theta.append(theta_3)

    prob_010_q2q1q0 = np.linalg.norm(input_data[2])/np.linalg.norm(input_data[2:4])
    theta_4 = 2*np.arccos(prob_010_q2q1q0)
    theta.append(theta_4)

    prob_100_q2q1q0 = np.linalg.norm(input_data[4])/np.linalg.norm(input_data[4:6])
    theta_5 = 2*np.arccos(prob_100_q2q1q0)
    theta.append(theta_5)

    prob_110_q2q1q0 = np.linalg.norm(input_data[6])/np.linalg.norm(input_data[6:8])
    theta_6 = 2*np.arccos(prob_110_q2q1q0)
    theta.append(theta_6)

    return theta


def gqhan():
    qc = amplitude_encoding()
    qc.barrier()
    qc = flexible_oracle(qc)
    qc = adaptive_diffusion(qc)
    return qc


def ansatz(num_qubits = 4):
    qc = QuantumCircuit(num_qubits)
    qc = flexible_oracle(qc)
    qc = adaptive_diffusion(qc)
    return qc


#def small_flexible_oracle(qc):
#    theta = ParameterVector('beta', length = 8)
#
#    #|000>
#    qc.rx(theta[0], 0)
#    qc.append(controlled_gate_000, [0,1,2,3])
#
#    #|001>
#    #qc.rx(theta[1], 0)
#    qc.append(controlled_gate_001, [0,1,2,3])
#
#    #|010>
#    #qc.rx(theta[2], 0)
#    qc.append(controlled_gate_010, [0,1,2,3])
#    
#    #|011>
#    #qc.rx(theta[3], 0)
#    qc.append(controlled_gate_011, [0,1,2,3])
#
#    #|100>
#    qc.rx(theta[4], 0)
#    qc.append(controlled_gate_100, [0,1,2,3])
#    
#    #|101>
#    #qc.rx(theta[5],0)
#    qc.append(controlled_gate_101, [0,1,2,3])
#    
#    #|110>
#    #qc.rx(theta[6],0)
#    qc.append(controlled_gate_110, [0,1,2,3])
#    
#    #|111>
#    #qc.rx(theta[7],0)
#    qc.append(controlled_gate_111, [0,1,2,3])
#
#    return qc
#
#
#def small_gqhan():
#    qc = amplitude_encoding()
#    qc.barrier()
#    qc = small_flexible_oracle(qc)
#    qc = adaptive_diffusion(qc)
#    return qc
#
#
#
#####DON'T UNCOMMENT USED ONLY TO TEST THE CODE
##qc_test = QuantumCircuit(4)  # Deve avere il numero corretto di qubit
##qc_test.x(0)
##qc_test.h([1,2,3])
##qc_test.append(controlled_gate_011, [0, 1, 2, 3])
##diffuser(qc_test)
##qc_test.measure_all()
##
##from data_preprocessing import PCA_data
##train_images_pca, test_images_pca, train_labels, test_labels = PCA_data()
##
##
##initial_state = train_images_pca[0]
##angles = prepare_angles(initial_state)
##qc_test = amplitude_encoding()
##qc_test = qc_test.assign_parameters(angles)
##print(initial_state)
##print(np.dot(initial_state,initial_state))
##print(qc_test.draw('text'))
#
##from qiskit import QuantumCircuit, transpile
##from qiskit_aer import AerSimulator
### Transpile for simulator
##simulator = AerSimulator()
##circ = transpile(qc_test, simulator)
##
#### Run and get counts
##result = simulator.run(circ).result()
##counts = result.get_counts(circ)
##
##import matplotlib.pyplot as plt
##from qiskit.visualization import plot_histogram
##plot_histogram(counts)
##plt.show()
##