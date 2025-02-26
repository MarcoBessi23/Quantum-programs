import numpy as np
from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit

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

def control_lambda(control_qubits:str):
    xcir = QuantumCircuit(1)
    xcir.x(0)
    xgate = xcir.to_gate(label='X').control(3, ctrl_state= control_qubits)
    
    return xgate    

def flexible_oracle(qc):
    theta = ParameterVector('theta', length = 8)

    #|000>
    qc.rx(theta[0], 0)
    qc.x(3)#put third bit in 1, same idea as for grover exercise in grover.py 
    qc.h(3)
    control0 = control_lambda('001')    
    qc.append(control0, [0,1,2,3])
    qc.h(3)
    qc.x(3)

    #|001>
    qc.rx(theta[1], 0)
    qc.h(3)
    qc.append(control0, [0,1,2,3])
    qc.h(3)

    #|010>
    qc.rx(theta[2], 0)
    
    qc.x(3)
    qc.h(3)
    control2 = control_lambda('101')    
    qc.append(control2, [0,1,2,3])
    qc.h(3)
    qc.x(3)
    
    #|011>
    qc.rx(theta[3], 0)
    qc.h(3)
    qc.append(control2, [0,1,2,3])
    qc.h(3)
    
    #|100>
    qc.rx(theta[4], 0)
    qc.x(3)
    qc.h(3)
    control3 = control_lambda('011')    
    qc.append(control3, [0,1,2,3])
    qc.h(3)
    qc.x(3)

    #|101>
    qc.rx(theta[5],0)
    qc.h(3)
    qc.append(control3, [0,1,2,3])
    qc.h(3)

    #|110>
    qc.rx(theta[6],0)
    qc.x(3)
    qc.h(3)
    control4 = control_lambda('111')
    qc.append(control4, [0,1,2,3])
    qc.h(3)
    qc.x(3)

    #|111>
    qc.rx(theta[7],0)
    qc.h(3)
    qc.append(control4, [0,1,2,3])
    qc.h(3)

    return qc

def adaptive_diffusion(qc):
    ''' 
    psi is the training parameter
    '''

    ### FIRST COSTRUCT THE MULTICONTROL ZGATE TO USE IN THE NETWORK
    zcir = QuantumCircuit(1)
    zcir.z(0)
    zgate = zcir.to_gate(label='z').control(3, ctrl_state= '111')


    psi = ParameterVector('psi', length = 6)
    qc.h([1,2,3])
    qc.cry(psi[0], 1, 2)
    qc.cry(psi[1], 2, 3)
    qc.cry(psi[2], 3, 1)
    ###append the multicontrol Z gate
    qc.append(zgate, [0,1,2,3])
    qc.cry(psi[3], 3, 1)
    qc.cry(psi[4], 2, 3)
    qc.cry(psi[5], 1, 2)

    qc.h([1,2,3])

    return qc

def GQHAN(input): #, theta, psi):
    '''
    Grover inspired Quantum Attention Network from https://arxiv.org/abs/2401.14089
    '''
    qc = amplitude_encoding(input)
    qc = flexible_oracle(qc)
    qc = adaptive_diffusion(qc)

    return qc