## Written by Eliott Rosenberg in 2021. If this is useful for you, please include me in your acknowledgments.


import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler import PassManager

rng = np.random.default_rng()

def random_compile(qc):
    def apply_padded_cx(qc,qubits,type='random'):
        if type == 'random':
            type = rng.integers(16)
        if type == 0:
            qc.cx(qubits[0],qubits[1])
        elif type == 1:
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[1])
        elif type == 2:
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
            qc.y(qubits[1])
        elif type == 3:
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
            qc.z(qubits[1])
        elif type == 4:
            qc.y(qubits[0])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
            qc.x(qubits[1])
        elif type == 5:
            qc.y(qubits[0])
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
        elif type == 6:
            qc.y(qubits[0])
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
            qc.z(qubits[1])
        elif type == 7:
            qc.y(qubits[0])
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
            qc.y(qubits[1])
        elif type == 8:
            qc.x(qubits[0])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
            qc.x(qubits[1])
        elif type == 9:
            qc.x(qubits[0])
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.x(qubits[0])
        elif type == 10:
            qc.x(qubits[0])
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
            qc.z(qubits[1])
        elif type == 11:
            qc.x(qubits[0])
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[0])
            qc.y(qubits[1])
        elif type == 12:
            qc.z(qubits[0])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
        elif type == 13:
            qc.z(qubits[0])
            qc.x(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[0])
            qc.x(qubits[1])
        elif type == 14:
            qc.z(qubits[0])
            qc.y(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.y(qubits[1])
        elif type == 15:
            qc.z(qubits[0])
            qc.z(qubits[1])
            qc.cx(qubits[0],qubits[1])
            qc.z(qubits[1])
        return qc
    if qc.num_clbits > 0:
        qc2 = QuantumCircuit(qc.num_qubits,qc.num_clbits)
    else:
        qc2 = QuantumCircuit(qc.num_qubits)
    for gate in qc:
        if gate[0].name == 'cx':
            # pad cx gate with 1-qubit gates.
            qc2 = apply_padded_cx(qc2,gate[1])
        else:
            if qc.num_clbits > 0:
                qc2.append(gate[0],gate[1],gate[2])
            else:
                qc2.append(gate[0],gate[1])
    return simplify(qc2)



def simplify(qc):
    p = Optimize1qGatesDecomposition(basis=['rz','sx','x','cx'])
    pm = PassManager(p)
    return pm.run(qc)


def fold(scale_factor,qc):
    max_scale = math.ceil((scale_factor - 1)/2)*2 + 1
    p = 1 - (max_scale - scale_factor)/3
    if qc.num_clbits > 0:
        qc2 = QuantumCircuit(qc.num_qubits,qc.num_clbits)
    else:
        qc2 = QuantumCircuit(qc.num_qubits)
    for gate in qc:
        if gate[0].name == 'cx':
            scale = rng.choice([max_scale,max_scale-2],p=[p,1-p])
            for _ in range(scale):
                qc2.cx(gate[1][0],gate[1][1])
        else:
            if qc.num_clbits > 0:
                qc2.append(gate[0],gate[1],gate[2])
            else:
                qc2.append(gate[0],gate[1])
    return qc2
    

def fold_and_compile(scale_factor,qc,rand_compile=True):
    qc = fold(scale_factor,qc)
    if rand_compile:
        qc = random_compile(qc)
    return qc