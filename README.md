# Overview
This code allows you to reproduce the experiments performed by Rosenberg et al (2021). Specifically, it contains functions to optimize variational ansatze both classically and hybridly, as well as functions to implement a variety of error mitigation techniques.

# To cite
To cite this code, cite the corresponding paper and link to this URL. If you have any questions or comments about this code, please contact Eliott Rosenberg at enr27--at--cornell--dot--edu.

# Dependencies
`qiskit`

`numpy`

`scipy`

`cupy` (optional; only if you want to use a GPU to perform classical evaluations)

`qiskit-aer-gpu` (optional; only for qiskit aer simulations using the gpu)


# Getting started
First, install the dependencies.

If you want to run a Variational Quantum Eigensolver (VQE) experiment, first modify `optimize_machine(...)` in `optimization.py` so that it saves the measured energy (E) and the classically evaluated energy (E_exact) to a file of your choice at each evaluation. Then modify `VQE.py` to your situation and execute `VQE.py`.

If you want to test the various error mitigation techniques described in Rosenberg et al (2021), follow the directions given as comments in `error_mitigation.py`. Search this file and `optimization.py` for `/your_directory/` and replace with the directory where you want to save data.



# General organization of file contents:

`VQE.py` can be executed to run a Variational Quantum Eigensolver (VQE) that uses measurements from actual quantum computers to tune an ansatz to the ground state of the Mixed-Field Ising Model. It should first be edited to suit your needs, and you may want to edit the cost function to save the measured energy at each evaluation. It can be easily modified to use a different Hamiltonian.


`optimization.py` contains functions that implement classical or hybrid optimization of variational ansatze. `optimize_machine(...)` is the function called in `VQE.py` that performs hybrid classical-quantum optimization. optimize(...) performs a classical optimization. `optimize_symm(...)` performs classical optimization, imposing permutation symmetry on the ansatz. See the file contents for more details.


`energy_evaluation.py` contains functions that submit ansatz circuits to IBM quantum computers and that extract the measured energies from the results. See the file contents for more details.


`error_mitigation.py` contains (as comments) human-readable instructions for benchmarking various error mitigation techniques. It also contains functions that implement these various error mitigation techniques.


`library.py` contains functions that are called by functions in the other files. (Although some functions in `energy_evaluation.py` are called in `error_mitigation.py`, etc.)
