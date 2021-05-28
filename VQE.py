## To reproduce Figs 4-5, run the following, and plot the results. You may want to edit optimize_machine in optimization.py so that it saves each energy evaluation to a file (save E and E_exact). (Note that two energy evaluations are performed in a single job, so E and E_exact have two components each.)

## If your computer has a gpu, you can set gpu=True to perform the classical evaluation using your gpu. Doing so requires cupy.

from optimization import optimize_machine
n = 20
hx = 1.5
hz = 0.1
backend_name = 'ibmq_sydney' # or ibmq_toronto for Fig 5
for l in range(3,7):
	optimize_machine(backend_name,n,l,hx,hz,shots=8192,readout_mitigate=False,symm=True,gpu=False)