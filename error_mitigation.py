import numpy as np
from library import *
from datetime import datetime
## To reproduce Figures 6-9, do the following:

################# Step 1 ###############################
## First, run the classical optimization. For example:
def optimize_all(n=12,hz=0.1):
	from optimization import optimize
	for l in [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]:
		for hx in [0.1,0.2,0.3,0.4,0.5,1.5]:
			optimize(n,l,hx,hz,method='BFGS',gpu=True,jac=True)
		
## if jac=True, the gradient is computed analytically, in parallel. If gpu=True, then the optimization uses a gpu. (This requires cupy.) If you do not have a gpu, you should set gpu=False. You can play around with the different optimization methods in https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.htm. We usually used TNC, but BFGS may work well too. In optimization.py, change /your_directory/... to be the location where you want to save parameters. You will need to set up the directories before running the optimizer.

## if you want to impose cyclic permutation symmetry, then instead do

def optimize_all_symm(n=12,hz=0.1):
	from optimization import optimize_symm
	for l in [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]:
		for hx in [0.1,0.2,0.3,0.4,0.5,1.5]:
			optimize_symm(n,l,hx,hz,method='BFGS',gpu=True,jac=True)
		
## Same comments about gpu and jac as above.

## Make sure that your optimizers have (approximately) converged before continuing. For our work, we imposed permutation symmetry for the 20 qubit case above 10 layers.




############# Step 2 #########################
## Next, to compare the optimization methods, submit ansatz circuits with the optimized parameters, in addition to \theta=0 circuits.


# use something like the following. It will need to be modified for your directory and whether you imposed permutation symmetry
def submit_saved_params(n,l,hx,backend_name,hz=0.1):
	from energy_evaluation import submit_ising, submit_ising_symm
	
	my_directory = '/your_directory/'
	if not hasattr(l,'__iter__'):
		l = [l]
	if not hasattr(hx,'__iter__'):
		hx = [hx]
	
	for li in l:
		if n < 20 or (n == 20 and li < 10):
			symm = False
			base_dir = my_directory+'ising/ALAy_cx/'
		elif n == 20 and li >= 10:
			symm = True
			base_dir = my_directory+'ising/ALAy_symm/'
		for hxi in hx:
			E = float(np.genfromtxt(base_dir+'n'+str(n)+'_l'+str(li)+'_hx'+str(hxi)+'_hz'+str(hz)+'/E.csv'))
			theta = np.genfromtxt(base_dir+'n'+str(n)+'_l'+str(li)+'_hx'+str(hxi)+'_hz'+str(hz)+'/theta.csv',delimiter=',')
			if not symm:
				submit_ising(n,theta,backend_name,shots=1024,hx=hxi,hz=hz,E=E)
			elif symm:
				submit_ising_symm(n,theta,backend_name,shots=8192,hx=hxi,hz=hz,E=E,input_condensed_theta=False)
				
def submit_zero_calibration(n,l,backend_name):
	from energy_evaluation import all_ising_Paulis_symm, submit_circuits
	
	if not hasattr(l,'__iter__'):
		l = [l]
	
	whichPauli = all_ising_Paulis_symm(n)
	for li in l:
		theta = np.zeros(n*(li+1))
		submit_circuits(theta,whichPauli,backend_name,tags=['zero_theta_calibration'],shots=8192)

## It is important that the backend is not recalibrated between when any of the above jobs run. To check the latest calibration datetime use

def latest_calibration_date(backend_name,n):
	from qiskit import IBMQ
	from energy_evaluation import load_qubit_map
	account = IBMQ.load_account()
	backend = account.get_backend(backend_name)
	properties = backend.properties()
	
	gates = properties.gates
	qubits = properties.qubits
	loop_qubits = load_qubit_map(backend_name,n)
	sx_dates = [gate.parameters[0].date for gate in gates if gate.gate == 'sx' and gate.qubits[0] in loop_qubits]
	cx_dates = [gate.parameters[0].date for gate in gates if gate.gate == 'cx' and gate.qubits[0] in loop_qubits and gate.qubits[1] in loop_qubits]
	em_dates = [ qubits[q][4].date for q in loop_qubits]
	
	return max( max(cx_dates), max(sx_dates), max(em_dates) )
	
## you might also want to check the latest calibration date at a time when a job ran. To do this, use:
def latest_calibration_date_from_job(job_id):
	from qiskit import IBMQ
	from energy_evaluation import load_qubit_map, read_from_tags
	job = account.backends.retrieve_job(job_id)
	n = read_from_tags('n',job.tags())
	properties = job.properties()
	backend_name = job.backend().name()
	
	gates = properties.gates
	qubits = properties.qubits
	loop_qubits = load_qubit_map(backend_name,n,0)
	sx_dates = [gate.parameters[0].date for gate in gates if gate.gate == 'sx' and gate.qubits[0] in loop_qubits]
	cx_dates = [gate.parameters[0].date for gate in gates if gate.gate == 'cx' and gate.qubits[0] in loop_qubits and gate.qubits[1] in loop_qubits]
	em_dates = [ qubits[q][4].date for q in loop_qubits]
	
	return max( max(cx_dates), max(sx_dates), max(em_dates) )
	
	
	
############ Step 3 ###############
## Now that your circuits have run successfully, it is time to analyze the results.
## Use the following functions to compare the observed damping factors to the predicted damping factors.

# The observed damping factor for a given job, with or without readout error mitigation applied, is 
def damping_from_job(job,readout_mitigate=True,readout_calibration_job=[]):
	from energy_evaluation import read_from_tags, ising_energy_from_job
	E_exact = read_from_tags('E',job.tags())
	E_meas, dE_meas = ising_energy_from_job(job,readout_mitigate,readout_calibration_job)
	print('E_meas = '+str(E_meas))
	print('dE_meas = '+str(dE_meas))
	damping = E_meas/E_exact
	d_damping = abs(dE_meas/E_exact)
	return damping, d_damping

## We use several methods of predicting the damping factor:




# From the perturbative regime:
def damping_est_pert(job,readout_mitigate=True,calibration_job=[]):
	from energy_evaluation import read_from_tags
	backend = job.backend()
	tags = job.tags()
	n = read_from_tags('n',tags)
	hz = read_from_tags('hz',tags)
	symm = 'symm' in tags
	l = read_from_tags('l',tags)
	
	hx_pert = [0.1,0.2,0.3,0.4,0.5]
	damping_all = []
	for hx in hx_pert:
		desired_tags = ['Ising','l = '+str(l),'hx = '+str(hx),'n = '+str(n),'hz = '+str(hz)]
		if symm:
			desired_tags.append('symm')
		job_pert = backend.jobs(limit=1,job_tags=desired_tags,job_tags_operator='AND')[0]
		damping_i, d_damping_i = damping_from_job(job_pert,readout_mitigate,calibration_job)
		damping_all.append(damping_i)
		
	damping = np.mean(damping_all)
	d_damping = np.std(damping_all)/np.sqrt(len(hx_pert))
	
	return damping, d_damping
	
	
	
	
	
	
# from small l:

def exp_fit(l,A,b):
	return A*np.exp(-b*l)

def small_l_fit(backend,n,hx,hz,max_l=15,readout_mitigate=True):
	from scipy.optimize import curve_fit
	l_all = range(0,max_l+1)
	damping_all = []
	d_damping_all = []
	for l in l_all:
		desired_tags = ['Ising','l = '+str(l),'hx = '+str(hx),'n = '+str(n),'hz = '+str(hz)]
		job = backend.jobs(limit=1,job_tags=desired_tags,job_tags_operator='AND')[0]
		damping_i, d_damping_i = damping_from_job(job_pert,readout_mitigate)
		damping_all.append(damping_i)
		d_damping_all.append(d_damping_i)
		
	fit_shallow = curve_fit(exp_fit,l_all,damping_all,p0=[1,0.5],sigma=d_damping_all,absolute_sigma=True)
	
	return fit_shallow
	
def pred_from_fit(l,fit,size=100000):
	rng = np.random.default_rng()
	params = rng.multivariate_normal(fit[0],fit[1],size=size)
	est = exp_fit(l,params[:,0],params[:,1])
	return [np.mean(est), np.std(est)]






# from zero theta calibration:


def Minv_uncorrelated_uncertainty(e0_0_pop, e1_0_pop, e0_1_pop, e1_1_pop, shots):
	num_trials = 100000
	rng = np.random.default_rng()
	Minv = []
	for trial in range(num_trials):
		e0_0 = rng.binomial(shots,e0_0_pop)/shots
		e1_0 = rng.binomial(shots,e1_0_pop)/shots
		e0_1 = rng.binomial(shots,e0_1_pop)/shots
		e1_1 = rng.binomial(shots,e1_1_pop)/shots
		M = [[ (1 - e0_0)*(1-e0_1), e1_0*(1-e0_1), e1_1*(1-e0_0), e1_0*e1_1], \
					 [e0_0*(1-e0_1), (1-e1_0)*(1-e0_1), e0_0*e1_1, e1_1*(1-e1_0)], \
					 [e0_1*(1-e0_0), e1_0*e0_1, (1-e0_0)*(1-e1_1), (1-e1_1)*e1_0], \
					 [e0_1*e0_0, (1-e1_0)*e0_1, e0_0*(1-e1_1), (1-e1_1)*(1-e1_0)]]
		Minv.append( np.linalg.inv(M))
	
	return np.mean(Minv,axis=0), np.std(Minv,axis=0)

def damping_from_zero_theta_fidelity(zero_calib_job,readout_mitigate=True,readout_calibration_job=None):
	# zero_calib_job should be the \theta=0 calibration job.
	from energy_evaluation import read_from_tags, load_qubit_map
	
	
	result = zero_calib_job.result()
	shots = result.to_dict().get('results')[0].get('shots')
	counts = result.get_counts()
	configs_all = read_from_tags('configs',zero_calib_job.tags())
	num_configs = len(configs_all[0])
	n = read_from_tags('n',zero_calib_job.tags())
	qubit_map = load_qubit_map(zero_calib_job.backend().name(),n)
	properties = zero_calib_job.properties()
	
	
	if readout_calibration_job == None:
		includes_one_qubit = True
	
	# readout mitigation:
	if readout_mitigate:
		if readout_calibration_job != None:
			qubits_measured_all = list(read_from_tags('qubits_measured_all',readout_calibration_job.tags()))
			qubits_measured_1 = [q for q in qubits_measured_all if len(q) == 1]
			qubits_measured_2 = [q for q in qubits_measured_all if len(q) == 2]
			includes_one_qubit = len(qubits_measured_1) > 0
			e_1qubit, Minv, de_1qubit, dMinv = analyze_readout_calibration_advanced(readout_calibration_job)
		elif readout_calibration_job == None:
			qubits_measured_1 = [ frozenset([i]) for i in range(n)]
			qubits_measured_2 = [ frozenset([i,(i+1)%n]) for i in range(n)]
			e_1qubit = np.array([ [properties.qubit_property(q,'prob_meas1_prep0')[0], properties.qubit_property(q,'prob_meas0_prep1')[0]] for q in qubit_map])
			de_1qubit = np.sqrt(e_1qubit*(1-e_1qubit)/5000)
			Minv = []
			dMinv = []
			for i in range(n):
				[e0_0, e1_0] = e_1qubit[i]
				[e0_1, e1_1] = e_1qubit[(i+1)%n]
				M = [[ (1 - e0_0)*(1-e0_1), e1_0*(1-e0_1), e1_1*(1-e0_0), e1_0*e1_1], \
					 [e0_0*(1-e0_1), (1-e1_0)*(1-e0_1), e0_0*e1_1, e1_1*(1-e1_0)], \
					 [e0_1*(1-e0_0), e1_0*e0_1, (1-e0_0)*(1-e1_1), (1-e1_1)*e1_0], \
					 [e0_1*e0_0, (1-e1_0)*e0_1, e0_0*(1-e1_1), (1-e1_1)*(1-e1_0)]]
				Minv.append( np.linalg.inv(M))
				Minv_mean, dMinv_i = Minv_uncorrelated_uncertainty(e0_0, e1_0, e0_1, e1_1, 5000)
				dMinv.append(dMinv_i)
		
		# ZZ0 term:
		C_zz0 = []
		dC_zz0 = []
		for which_config in range(num_configs):
			config = configs_all[0][which_config]
			if config >= 0:
				qubits_measured = (np.array([0,1]) + config)%n
			elif config < 0:
				qubits_measured = (-np.array([0,1]) + config + 1)%n
			which_calib_term = qubits_measured_2.index(frozenset(qubits_measured))
			reversed = not (qubits_measured[0] == list(qubits_measured_2[which_calib_term])[0])
			if reversed:
				counts_vector = [counts[which_config].get(bitstr,0) for bitstr in ['00','01','10','11'] ]
			else:
				counts_vector = [counts[which_config].get(bitstr,0) for bitstr in ['00','10','01','11'] ]
			counts_vector = np.array(counts_vector)
			d_counts_vector = np.sqrt(counts_vector * (shots - counts_vector)/shots)
			counts_mitigated = Minv[which_calib_term]@counts_vector
			d_counts_mitigated = np.sqrt( (dMinv[which_calib_term]**2)@(counts_vector**2) + Minv[which_calib_term]**2 @ d_counts_vector**2 )
			C_zz0.append( (counts_mitigated[0]/shots - 1/4)*4/3 )
			dC_zz0.append( d_counts_mitigated[0]/shots*4/3 )
			
			
		# ZZ1 term:
		C_zz1 = []
		dC_zz1 = []
		for which_config in range(num_configs):
			config = configs_all[1][which_config]
			if config >= 0:
				qubits_measured = (np.array([1,2]) + config)%n
			elif config < 0:
				qubits_measured = (-np.array([1,2]) + config + 1)%n
			which_calib_term = qubits_measured_2.index(frozenset(qubits_measured))
			reversed = not (qubits_measured[0] == list(qubits_measured_2[which_calib_term])[0])
			if reversed:
				counts_vector = [counts[num_configs+which_config].get(bitstr,0) for bitstr in ['00','01','10','11'] ]
			else:
				counts_vector = [counts[num_configs+which_config].get(bitstr,0) for bitstr in ['00','10','01','11'] ]
			counts_vector = np.array(counts_vector)
			d_counts_vector = np.sqrt(counts_vector * (shots - counts_vector)/shots)
			counts_mitigated = Minv[which_calib_term]@counts_vector
			d_counts_mitigated = np.sqrt( (dMinv[which_calib_term]**2)@(counts_vector**2) + Minv[which_calib_term]**2 @ d_counts_vector**2 )
			C_zz1.append( (counts_mitigated[0]/shots - 1/4)*4/3 )
			dC_zz1.append( d_counts_mitigated[0]/shots*4/3 )
			
		# Z0 term:
		C_z0 = []
		dC_z0 = []
		for which_config in range(num_configs):
			config = configs_all[4][which_config]
			if config >= 0:
				qubit_measured = config
			else:
				qubit_measured = (config+1)%n
			if includes_one_qubit:
				[e0,e1] = e_1qubit[ qubits_measured_1.index( frozenset([qubit_measured]))]
				[de0,de1] = de_1qubit[ qubits_measured_1.index( frozenset([qubit_measured]))]
			else:
				e0 = properties.qubit_property(qubit_map[qubit_measured],'prob_meas1_prep0')[0]
				e1 = properties.qubit_property(qubit_map[qubit_measured],'prob_meas0_prep1')[0]
				de0 = np.sqrt(e0*(1-e0)/5000)
				de1 = np.sqrt(e1*(1-e1)/5000)
			f0_measured = counts[4*num_configs+which_config].get('0',0)/shots
			f0 = (f0_measured - e1)/(1-e0-e1)
			d_f0_measured = np.sqrt(f0_measured*(1-f0_measured)/shots)
			df0 = np.sqrt( (d_f0_measured/(1-e0-e1))**2 + ( -1/(1-e0-e1) + (f0_measured - e1)/(1-e0-e1)**2)**2 * de1**2 + ((f0_measured - e1)/(1-e0-e1)**2 * de0)**2 )
			C_z0.append((f0-1/2)*2)
			dC_z0.append(2*df0)
		
		# Z1 term:
		C_z1 = []
		dC_z1 = []
		for which_config in range(num_configs):
			config = configs_all[5][which_config]
			if config >= 0:
				qubit_measured = (1+config)%n
			else:
				qubit_measured = (config)%n
			if includes_one_qubit:
				[e0,e1] = e_1qubit[ qubits_measured_1.index( frozenset([qubit_measured]))]
				[de0,de1] = de_1qubit[ qubits_measured_1.index( frozenset([qubit_measured]))]
			else:
				e0 = properties.qubit_property(qubit_map[qubit_measured],'prob_meas1_prep0')[0]
				e1 = properties.qubit_property(qubit_map[qubit_measured],'prob_meas0_prep1')[0]
				de0 = np.sqrt(e0*(1-e0)/5000)
				de1 = np.sqrt(e1*(1-e1)/5000)
			f0_measured = counts[5*num_configs+which_config].get('0',0)/shots
			f0 = (f0_measured - e1)/(1-e0-e1)
			d_f0_measured = np.sqrt(f0_measured*(1-f0_measured)/shots)
			df0 = np.sqrt( (d_f0_measured/(1-e0-e1))**2 + ( -1/(1-e0-e1) + (f0_measured - e1)/(1-e0-e1)**2)**2 * de1**2 + ((f0_measured - e1)/(1-e0-e1)**2 * de0)**2 )
			C_z1.append((f0-1/2)*2)
			dC_z1.append(2*df0)
	
	else:
		C_zz0 = [ ( counts[i].get('00',0)/shots - 1/4)*4/3 for i in range(num_configs) ]
		C_zz1 = [ ( counts[i].get('00',0)/shots - 1/4)*4/3 for i in range(num_configs,2*num_configs)]
		C_z0 = [ ( counts[i].get('0',0)/shots - 1/2)*2 for i in range(4*num_configs,5*num_configs) ]
		C_z1 = [ ( counts[i].get('0',0)/shots - 1/2)*2 for i in range(5*num_configs,6*num_configs) ]

		
		dC_zz0 = [ 4/3*np.sqrt( counts[i].get('00',0)/shots * (1 - counts[i].get('00',0)/shots) / shots ) for i in range(num_configs) ]
		dC_zz1 = [ 4/3*np.sqrt( counts[i].get('00',0)/shots * (1 - counts[i].get('00',0)/shots) / shots ) for i in range(num_configs,2*num_configs) ]
		dC_z0 = [ 2*np.sqrt( counts[i].get('0',0)/shots * (1 - counts[i].get('0',0)/shots) / shots ) for i in range(4*num_configs,5*num_configs) ]
		dC_z1 = [ 2*np.sqrt( counts[i].get('0',0)/shots * (1 - counts[i].get('0',0)/shots) / shots ) for i in range(5*num_configs,6*num_configs) ]
	
	compensation_factors_list = [C_zz0, C_zz1, C_z0, C_z1, C_z0, C_z1]
	d_compensation_factors_list = [dC_zz0, dC_zz1, dC_z0, dC_z1, dC_z0, dC_z1]
	
	compensation_factors = C_zz0 + C_zz1 + C_z0 + C_z1
	d_compensation_factors = dC_zz0 + dC_zz1 + dC_z0 + dC_z1
	
	print('compensation factors = '+str(compensation_factors))
	print('d_compensation factors = '+str(d_compensation_factors))

	damping = np.mean(compensation_factors)
	d_damping = np.sqrt(np.sum(np.square(d_compensation_factors)))/len(compensation_factors)
	
	return damping, d_damping

	
def damping_from_zero_theta_energy(zero_calib_job,hx,hz,readout_mitigate=True,readout_calibrate_job=[]):
	from energy_evaluation import read_from_tags, ising_energy_from_job, energy_from_job
	E_exact = -2*(1+hz)
	coeffs = [-1 for _ in range(2)] + [-hx for _ in range(2)] + [-hz for _ in range(2)]
	E_meas, dE_meas = energy_from_job(zero_calib_job,coeffs,readout_mitigate,readout_calibrate_job)
	damping = E_meas/E_exact
	d_damping = abs(dE_meas/E_exact)
	return damping, d_damping






# finally, we have the two methods which estimate the damping from the reported error rates


# simulating with qiskit aer noise model (not scalable):

def noise_model_from_properties(properties,include_gate_errors=True,include_readout_errors=True):
	from qiskit.providers.aer.noise import device, NoiseModel
	gates = properties.gates
	basis_gates = list({g.gate for g in gates})
	noise_model = NoiseModel(basis_gates=basis_gates)
	
	if include_gate_errors:
		gate_errors = device.basic_device_gate_errors(properties)
		for gate_error in gate_errors:
			noise_model.add_quantum_error(gate_error[2],gate_error[0],gate_error[1])
	
	if include_readout_errors:
		readout_errors = device.basic_device_readout_errors(properties)
		for readout_error in readout_errors:
			noise_model.add_readout_error(readout_error[1], readout_error[0])

	return noise_model


def simulate_job(job,include_noise=True,gpu=True,include_gate_errors=True,include_readout_errors=True,density_matrix=True):
	# the gpu option requires qiskit-aer-gpu
	from qiskit import QuantumCircuit, execute, Aer, IBMQ
	import qiskit.providers.aer.noise as noise
	from qiskit.providers.aer import QasmSimulator
	from energy_evaluation import ansatz_circuit, load_qubit_map, read_from_tags, cycle_QuantumCircuit, energy_from_counts
	backend = job.backend()
	machine = backend.name()
	tags = job.tags()
	n = read_from_tags('n',tags)
	hx = read_from_tags('hx',tags)
	hz = read_from_tags('hz',tags)
	E = read_from_tags('E',tags)
	l = read_from_tags('l',tags)
	paulis = read_from_tags('whichPauli',tags)
	configs = read_from_tags('configs',tags)
	theta = read_from_tags('theta',tags)
	symm = 'symm' in tags

	if include_noise:
		#noise_model = noise.NoiseModel.from_backend(backend)
		noise_model = noise_model_from_properties(job.properties(),include_gate_errors,include_readout_errors)
	
		# Get coupling map from backend
		coupling_map = backend.configuration().coupling_map
	
		# Get basis gates from noise model
		basis_gates = noise_model.basis_gates
		

		if gpu and density_matrix:
			simulator = QasmSimulator(method='density_matrix_gpu')
		elif not gpu and density_matrix:
			simulator = QasmSimulator(method='density_matrix',max_parallel_threads=30)
		elif gpu and not density_matrix:
			simulator = QasmSimulator(method='statevector_gpu')
		elif not gpu and not density_matrix:
			simulator = QasmSimulator(method='statevector')

	qubits0 = load_qubit_map(machine,n)
	qc = []
	multi_theta = len( np.shape(theta) ) > 1
	if not multi_theta:
		theta = [theta]
	for theta_i in theta:
		for i in range(len(paulis)):
			pauli = paulis[i]
			for config in configs[i]:
				qc_i = ansatz_circuit(theta_i,pauli)
				qc_i = cycle_QuantumCircuit(qc_i,config)
				qc.append(qc_i)
	if include_noise:
		job2 = execute(qc, simulator, basis_gates=basis_gates, noise_model=noise_model,coupling_map=coupling_map,initial_layout=qubits0)
	else:
		job2 = execute(qc, Aer.get_backend('qasm_simulator'))
	
	
	counts = job2.result().get_counts()
	if symm:
		coeffs = [-1 for _ in range(2)] + [-hx for _ in range(2)] + [-hz for _ in range(2)]
		coeffs = np.array(coeffs) * n//2
	else:
		coeffs = [-1 for _ in range(n)] + [-hx for _ in range(n)] + [-hz for _ in range(n)]
	
	E, dE = energy_from_counts(counts,coeffs)
	
	return E, dE



def damping_from_aer_simulation(job,include_noise=True,gpu=True,include_gate_errors=True,include_readout_errors=True,density_matrix=True):
	# readout error is included in the aer simulation, so this should be compared to the measured dampings without readout error mitigation
	from energy_evaluation import read_from_tags
	E_exact = read_from_tags('E',job.tags())
	E_pred, dE_pred = simulate_job(job,include_noise,gpu,include_gate_errors,include_readout_errors,density_matrix)
	damping = E_pred/E_exact
	d_damping = dE_pred/E_exact
	return damping, d_damping



# multiplying fidelities:

def energy_from_job_mult_fidelities(job,coeffs):
	from library import damping_from_fidelities

	counts = job.result().get_counts()
	tags = job.tags()
	whichPauli_all = read_from_tags('whichPauli',tags)
	n = read_from_tags('n',tags)
	l = read_from_tags('l',tags)
	configs = read_from_tags('configs',tags)
	num_configs = len(configs[0])
	num_thetas = len(counts)//(num_configs*len(whichPauli_all))
	num_terms = len(whichPauli_all)
	
	multi_coeffs = len(np.shape(coeffs)) == 2
	if multi_coeffs:
		coeffs_all = coeffs
	
	backend_name = job.backend().name()
	qubits = load_qubit_map(backend_name,n)
	properties = job.properties()
	e0 = np.array([properties.qubits[q][6].value for q in qubits])
	e1 = np.array([properties.qubits[q][5].value for q in qubits])
	em = (e0+e1)/2
	e1_minus_e0 = e1 - e0
	
	e_cx = [properties.gate_error('cx',[qubits[i],qubits[(i+1)%n]]) for i in range(n)]
	e_sx = [properties.gate_error('sx',q) for q in qubits]
	
		
	E_all = []
	dE_all = []
	
	for which_theta in range(num_thetas):
		E = 0
		dE2 = 0
		if multi_coeffs:
			coeffs = coeffs_all[which_theta]
		
		
		for term in range(num_terms):
			whichPauli = whichPauli_all[term]
			qubits_measured = np.array([i for i in range(n) if whichPauli[i]>0])
			for which_config in range(num_configs):
				config = configs[term][which_config]
				if config >= 0:
					qubits_measured_config = np.mod(qubits_measured + config, n)
				elif config < 0:
					qubits_measured_config = np.mod( -qubits_measured + config + 1, n)
				
				P,dP = P_from_counts(counts[which_theta*num_configs*num_terms + num_configs*term + which_config])
				
				P,dP = readout_error_correct(P,dP,em[qubits_measured_config],e1_minus_e0[qubits_measured_config])
				
				predicted_damping = damping_from_fidelities(l,whichPauli, e_cx, e_sx,config)
				
				P = P/predicted_damping
				dP = dP/predicted_damping
			
				E += coeffs[term] * P /num_configs
				dE2 += (coeffs[term] * dP /num_configs )**2
		E_all.append(E)
		dE_all.append(np.sqrt(dE2))
	
	if num_thetas > 1:
		return E_all, dE_all
	elif num_thetas == 1:
		return E_all[0], dE_all[0]





def ising_energy_from_job_mult_fidelities(job):
	tags = job.tags()
	symm = 'symm' in tags
	hx = read_from_tags('hx',tags)
	hz = read_from_tags('hz',tags)
	n = read_from_tags('n',tags)
	if symm: # symmetric ansatz
		m = 2
	else:
		m = n
	multi_hx = hasattr(hx,'__iter__')
	if not multi_hx:
		coeffs = [-1 for _ in range(m)] + [-hx for _ in range(m)] + [-hz for _ in range(m)]
	elif multi_hx:
		coeffs = [[-1 for _ in range(m)] + [-hxi for _ in range(m)] + [-hz for _ in range(m)] for hxi in hx]
		
	if symm:
		coeffs = np.array(coeffs) * n//2 # rescale the coefficients
		
	return energy_from_job_mult_fidelities(job,coeffs)


def damping_mult_fidelities(job):
	# this is the damping factor including readout errors
	from energy_evaluation import ising_energy_from_job
	E_meas, dE_meas = ising_energy_from_job(job)
	E_mitigated, dE_mitigated = ising_energy_from_job_mult_fidelities(job)
	
	damping = E_meas/E_mitigated
	
	return damping
	
	


##### for calibrating readout error mitigation:

def qubits_measured_from_job(job):
	from energy_evaluation import read_from_tags
	tags = job.tags()
	whichPauli_all = read_from_tags('whichPauli',tags)
	configs_all = read_from_tags('configs',tags)
	n = read_from_tags('n',tags)
	qubits_measured_all = set()
	num_terms = len(whichPauli_all)
	for term in range(num_terms):
		whichPauli = whichPauli_all[term]
		configs = configs_all[term]
		qubits_measured_0 = np.array([i for i in range(n) if whichPauli[i] > 0])
		for config in configs:
			if config >= 0:
				qubits_measured = (qubits_measured_0 + config) % n
			else:
				qubits_measured = (-qubits_measured_0 + config + 1) % n
			qubits_measured_all.add( frozenset(qubits_measured) )
	return qubits_measured_all




def submit_readout_calibration_circuits(n,backend,qubits_measured_all,shots=8192):
	from qiskit import QuantumCircuit, execute
	from energy_evaluation import load_qubit_map
	qc_all = []
	qubits = load_qubit_map(backend.name(),n)
	for qubits_measured in qubits_measured_all:
		qubits_measured = list(qubits_measured)
		if len(qubits_measured) == 2:
			for x0 in [False, True]:
				for x1 in [False, True]:
					qc = QuantumCircuit(n,2)
					if x0:
						qc.x(qubits_measured[0])
					if x1:
						qc.x(qubits_measured[1])
					qc.measure(qubits_measured[0],0)
					qc.measure(qubits_measured[1],1)
					qc_all.append(qc)
		elif len(qubits_measured) == 1:
			for x0 in [False,True]:
				qc = QuantumCircuit(n,1)
				if x0:
					qc.x(qubits_measured[0])
				qc.measure(qubits_measured[0],0)
				qc_all.append(qc)
			
				
	job =  execute(qc_all, backend=backend, shots=shots, initial_layout=qubits, job_tags=['readout_calibration','qubits_measured_all = '+str(qubits_measured_all)])
	
	return job
	
	

def submit_readout_calibration_datetimes(n,backend_name,start,end,shots=8192):
	from qiskit import IBMQ
	account = IBMQ.load_account()
	backend = account.get_backend(backend_name)
	jobs = backend.jobs(limit=1000,start_datetime=start,end_datetime=end,job_tags=['n = '+str(n)])
	qubits_measured_all = set()
	print('# jobs = '+str(len(jobs)))
	for job in jobs:
		qubits_measured_all = qubits_measured_all.union( qubits_measured_from_job(job) )
	
	print('qubits_measured_all = '+str(qubits_measured_all))
	
	return submit_readout_calibration_circuits(n,backend,qubits_measured_all,shots)



def analyze_readout_calibration(calibration_job):


	result = calibration_job.result()
	shots = result.results[0].shots
	counts = result.get_counts()
	num_pairs = len(counts)//4
	e0 = []
	e1 = []
	for pair in range(num_pairs):
		e0_pair = (counts[4*pair].get('01',0) + counts[4*pair].get('10',0) + counts[4*pair+3].get('01',0) + counts[4*pair+3].get('10',0))/(2*shots)
		
		e1_pair = (counts[4*pair+1].get('00',0) + counts[4*pair+1].get('11',0) + counts[4*pair+2].get('00',0) + counts[4*pair+2].get('11',0))/(2*shots)
		e0.append(e0_pair)
		e1.append(e1_pair)
		
	e0 = np.array(e0)
	e1 = np.array(e1)
		
	return e0, e1
	
	
def analyze_readout_calibration_advanced(calibration_job):


	from energy_evaluation import read_from_tags
	result = calibration_job.result()
	shots = result.results[0].shots
	counts = result.get_counts()
	qubits_measured_all = list(read_from_tags('qubits_measured_all',calibration_job.tags()))
	circuit = 0
	e_1qubit = []
	Minv = []
	dMinv = []
	for qubits_measured in qubits_measured_all:
		num_qubits = len(list(counts[circuit])[0])
		if num_qubits == 1:
			e_1qubit.append( [counts[circuit].get('1',0)/shots, counts[circuit+1].get('0',0)/shots]  )
			circuit += 2
		elif num_qubits == 2:
			M =  [ [ counts[circuit+j].get(bitstr,0)/shots for j in range(4)] for bitstr in ['00','10','01','11'] ]
			M = np.array(M)
			#dM = np.sqrt(M*(1-M)/shots)
			Minv_i_est, dMinv_i = uncertainty_in_Minv(M,shots)
			Minv_i = np.linalg.inv(M)
			
			
			Minv.append( Minv_i )
			dMinv.append(dMinv_i)
			circuit += 4
	
	e_1qubit = np.array(e_1qubit)
	de_1qubit = np.sqrt(e_1qubit * (1-e_1qubit) /shots)
	return e_1qubit, Minv, de_1qubit, dMinv
			

def uncertainty_in_Minv(M,shots):
	rng = np.random.default_rng()
	trials = 10000
	Minv = []
	for trial in range(trials):
		M_trial = (np.array([ rng.multinomial(shots,column) for column in M.T ]).T)/shots
		Minv.append(  np.linalg.inv(M_trial) )
		
	return np.mean(Minv,axis=0), np.std(Minv,axis=0)

# plotting:

def rel_error(damping,d_damping,damping_est,d_damping_est):
	damping = np.array(damping)
	damping_est = np.array(damping_est)
	d_damping = np.array(d_damping)
	d_damping_est = np.array(d_damping_est)
	rel_error = (damping - damping_est)/damping_est
	d_rel_error = np.sqrt( (d_damping/damping_est)**2 + (d_damping_est*damping/damping_est**2)**2 )
	return [rel_error,d_rel_error]
	
	
def rel_error_score(r,dr,th):
	# assigns 3, 2, 1, or 0
	score = []
	for i in range(len(r)):
		if -th < r[i] - dr[i] and r[i] + dr[i] < th:
			score.append(3)
		elif abs(r[i] + dr[i]) < th or abs(r[i] - dr[i]) < th or (r[i] + dr[i] > th and r[i] - dr[i] < - th):
			score.append(2)
		elif abs(r[i]) - th < 2*dr[i]:
			score.append(1)
		elif np.isnan(r[i]):
			score.append(np.nan)
		else:
			score.append(0)
	return score
	
def plot_from_machine_layers(n=20,hx=1.5,hz=0.1,start_date = datetime(year=2021,month=4,day=29,hour=0,minute=2,second=30),end_date = datetime(year=2021,month=4,day=30,hour=23,minute=0),backend_name='ibmq_toronto',readout_calibrate = True,threshold=0.1,load_saved=False):

	# n=20,hx=1.5,hz=0.1,start_date = datetime(year=2021,month=4,day=13,hour=0,minute=2,second=30),end_date = datetime(year=2021,month=4,day=13,hour=23,minute=0),backend_name='ibmq_toronto'

	import matplotlib.pyplot as plt
	from energy_evaluation import read_from_tags
	
	save_dir = '/your_directory/damping_factors/'+backend_name+'/n'+str(n)+'/'
	
	
	damping_readout = []
	d_damping_readout = []
	damping_readout_calibrate = []
	d_damping_readout_calibrate = []
	damping_raw = []
	d_damping_raw = []
	
	damping_pert_readout = []
	d_damping_pert_readout = []
	damping_pert_raw = []
	d_damping_pert_raw = []
	damping_pert_readout_calibrate = []
	d_damping_pert_readout_calibrate = []
	
	damping_zero_fid_readout = []
	d_damping_zero_fid_readout = []
	damping_zero_fid_raw = []
	d_damping_zero_fid_raw = []
	damping_zero_fid_readout_calibrate = []
	d_damping_zero_fid_readout_calibrate = []
	
	damping_zero_energy_readout = []
	d_damping_zero_energy_readout = []
	damping_zero_energy_raw = []
	d_damping_zero_energy_raw = []
	damping_zero_energy_readout_calibrate = []
	d_damping_zero_energy_readout_calibrate = []
	
	

	l = []
	
	
	if not load_saved:
		from qiskit import IBMQ
		account = IBMQ.load_account()
		backend = account.get_backend(backend_name)
		jobs = backend.jobs(limit=1000,start_datetime=start_date,end_datetime=end_date,job_tags=['hx = '+str(hx),'n = '+str(n),'hz = '+str(hz)],job_tags_operator='AND',status='DONE') + backend.jobs(limit=1000,start_datetime=start_date,end_datetime=end_date,job_tags=['hx = '+str([hx]),'n = '+str(n),'hz = '+str(hz)],job_tags_operator='AND',status='DONE')
		
		if readout_calibrate:
			readout_calibrate_job = backend.jobs(limit=1,start_datetime=start_date,end_datetime=end_date,job_tags=['readout_calibration'],status='DONE')[0]
		else:
			readout_calibrate_job = []
		
		
		for job in jobs:
			l.append(read_from_tags('l',job.tags()))
			print('l = '+str(l[-1]))
			if readout_calibrate:
				damping_readout_calibrate_i, d_damping_readout_calibrate_i = damping_from_job(job, True, readout_calibrate_job)
				damping_readout_calibrate.append(damping_readout_calibrate_i)
				d_damping_readout_calibrate.append(d_damping_readout_calibrate_i)
			damping_readout_i, d_damping_readout_i = damping_from_job(job, True)
			damping_readout.append(damping_readout_i)
			d_damping_readout.append(d_damping_readout_i)
			damping_raw_i, d_damping_raw_i = damping_from_job(job, False)
			damping_raw.append(damping_raw_i)
			d_damping_raw.append(d_damping_raw_i)
			
			
			
			if not (l[-1] == 10 and backend_name == 'ibmq_toronto' and n == 12 and start_date.month==4 and start_date.day==26):
				damping_pert_readout_i, d_damping_pert_readout_i = damping_est_pert(job,readout_mitigate=True)
				damping_pert_raw_i, d_damping_pert_raw_i = damping_est_pert(job,readout_mitigate=False)
			elif l[-1] == 10:
				damping_pert_readout_i = np.nan
				d_damping_pert_readout_i = np.nan
				damping_pert_raw_i = np.nan
				d_damping_pert_raw_i = np.nan
				damping_pert_readout_calibrate_i = np.nan
				d_damping_pert_readout_calibrate_i = np.nan
			damping_pert_raw.append(damping_pert_raw_i)
			d_damping_pert_raw.append(d_damping_pert_raw_i)
			damping_pert_readout.append(damping_pert_readout_i)
			d_damping_pert_readout.append(d_damping_pert_readout_i)
			if readout_calibrate and not (l[-1] == 10 and backend_name == 'ibmq_toronto' and n == 12 and start_date.month==4 and start_date.day==26):
				damping_pert_readout_calibrate_i, d_damping_pert_readout_calibrate_i = damping_est_pert(job,True,readout_calibrate_job)
			if readout_calibrate:
				damping_pert_readout_calibrate.append(damping_pert_readout_calibrate_i)
				d_damping_pert_readout_calibrate.append(d_damping_pert_readout_calibrate_i)
			
			
			zero_calib_job = backend.jobs(limit=1,start_datetime=start_date,end_datetime=end_date,job_tags=['zero_theta_calibration', 'l = '+str(l[-1]),'n = '+str(n)],job_tags_operator='AND',status='DONE')[0]
			damping_zero_fid_raw_i, d_damping_zero_fid_raw_i = damping_from_zero_theta_fidelity(zero_calib_job,False,None)
			damping_zero_fid_readout_i, d_damping_zero_fid_readout_i = damping_from_zero_theta_fidelity(zero_calib_job,True,None)
			if readout_calibrate:
				damping_zero_fid_readout_calibrate_i, d_damping_zero_fid_readout_calibrate_i = damping_from_zero_theta_fidelity(zero_calib_job,True,readout_calibrate_job)
				damping_zero_fid_readout_calibrate.append(damping_zero_fid_readout_calibrate_i)
				d_damping_zero_fid_readout_calibrate.append(d_damping_zero_fid_readout_calibrate_i)
			
			damping_zero_fid_raw.append(damping_zero_fid_raw_i)
			d_damping_zero_fid_raw.append(d_damping_zero_fid_raw_i)
			damping_zero_fid_readout.append(damping_zero_fid_readout_i)
			d_damping_zero_fid_readout.append(d_damping_zero_fid_readout_i)
			
			
			damping_zero_energy_raw_i, d_damping_zero_energy_raw_i = damping_from_zero_theta_energy(zero_calib_job,hx,hz,False,[])
			damping_zero_energy_readout_i, d_damping_zero_energy_readout_i = damping_from_zero_theta_energy(zero_calib_job,hx,hz,True,[])
			if readout_calibrate:
				damping_zero_energy_readout_calibrate_i, d_damping_zero_energy_readout_calibrate_i = damping_from_zero_theta_energy(zero_calib_job,hx,hz,True,readout_calibrate_job)
				damping_zero_energy_readout_calibrate.append(damping_zero_energy_readout_calibrate_i)
				d_damping_zero_energy_readout_calibrate.append(d_damping_zero_energy_readout_calibrate_i)

			damping_zero_energy_raw.append(damping_zero_energy_raw_i)
			d_damping_zero_energy_raw.append(d_damping_zero_energy_raw_i)
			damping_zero_energy_readout.append(damping_zero_energy_readout_i)
			d_damping_zero_energy_readout.append(d_damping_zero_energy_readout_i)
			
			
			
			
			
			
		l = np.array(l,dtype=np.int)
		
		
		
		np.savetxt(save_dir+'l.csv',l)
		np.savetxt(save_dir+'damping_readout.csv',damping_readout)
		np.savetxt(save_dir+'d_damping_readout.csv',d_damping_readout)
		np.savetxt(save_dir+'damping_readout_calibrate.csv',damping_readout_calibrate)
		np.savetxt(save_dir+'d_damping_readout_calibrate.csv',d_damping_readout_calibrate)
		np.savetxt(save_dir+'damping_raw.csv',damping_raw)
		np.savetxt(save_dir+'d_damping_raw.csv',d_damping_raw)
		
		np.savetxt(save_dir+'damping_pert_readout.csv',damping_pert_readout)
		np.savetxt(save_dir+'d_damping_pert_readout.csv',d_damping_pert_readout)
		np.savetxt(save_dir+'damping_pert_raw.csv',damping_pert_raw)
		np.savetxt(save_dir+'d_damping_pert_raw.csv',d_damping_pert_raw)
		np.savetxt(save_dir+'damping_pert_readout_calibrate.csv',damping_pert_readout_calibrate)
		np.savetxt(save_dir+'d_damping_pert_readout_calibrate.csv',d_damping_pert_readout_calibrate)
		
		np.savetxt(save_dir+'damping_zero_fid_readout.csv',damping_zero_fid_readout)
		np.savetxt(save_dir+'d_damping_zero_fid_readout.csv',d_damping_zero_fid_readout)
		np.savetxt(save_dir+'damping_zero_fid_raw.csv',damping_zero_fid_raw)
		np.savetxt(save_dir+'d_damping_zero_fid_raw.csv',d_damping_zero_fid_raw)
		np.savetxt(save_dir+'damping_zero_fid_readout_calibrate.csv',damping_zero_fid_readout_calibrate)
		np.savetxt(save_dir+'d_damping_zero_fid_readout_calibrate.csv',d_damping_zero_fid_readout_calibrate)
		
		np.savetxt(save_dir+'damping_zero_energy_readout.csv',damping_zero_energy_readout)
		np.savetxt(save_dir+'d_damping_zero_energy_readout.csv',d_damping_zero_energy_readout)
		np.savetxt(save_dir+'damping_zero_energy_raw.csv',damping_zero_energy_raw)
		np.savetxt(save_dir+'d_damping_zero_energy_raw.csv',d_damping_zero_energy_raw)
		np.savetxt(save_dir+'damping_zero_energy_readout_calibrate.csv',damping_zero_energy_readout_calibrate)
		np.savetxt(save_dir+'d_damping_zero_energy_readout_calibrate.csv',d_damping_zero_energy_readout_calibrate)
		
		
	elif load_saved:
		l = np.genfromtxt(save_dir+'l.csv')
		l = np.array(l,dtype=np.int)
		damping_readout = np.genfromtxt(save_dir+'damping_readout.csv')
		d_damping_readout = np.genfromtxt(save_dir+'d_damping_readout.csv')
		damping_readout_calibrate =np.genfromtxt(save_dir+'damping_readout_calibrate.csv')
		d_damping_readout_calibrate = np.genfromtxt(save_dir+'d_damping_readout_calibrate.csv')
		damping_raw = np.genfromtxt(save_dir+'damping_raw.csv')
		d_damping_raw = np.genfromtxt(save_dir+'d_damping_raw.csv')
		
		damping_pert_readout = np.genfromtxt(save_dir+'damping_pert_readout.csv')
		d_damping_pert_readout = np.genfromtxt(save_dir+'d_damping_pert_readout.csv')
		damping_pert_raw = np.genfromtxt(save_dir+'damping_pert_raw.csv')
		d_damping_pert_raw = np.genfromtxt(save_dir+'d_damping_pert_raw.csv')
		damping_pert_readout_calibrate = np.genfromtxt(save_dir+'damping_pert_readout_calibrate.csv')
		d_damping_pert_readout_calibrate = np.genfromtxt(save_dir+'d_damping_pert_readout_calibrate.csv')
	
		damping_zero_fid_readout = np.genfromtxt(save_dir+'damping_zero_fid_readout.csv')
		d_damping_zero_fid_readout = np.genfromtxt(save_dir+'d_damping_zero_fid_readout.csv')
		damping_zero_fid_raw = np.genfromtxt(save_dir+'damping_zero_fid_raw.csv')
		d_damping_zero_fid_raw = np.genfromtxt(save_dir+'d_damping_zero_fid_raw.csv')
		damping_zero_fid_readout_calibrate = np.genfromtxt(save_dir+'damping_zero_fid_readout_calibrate.csv')
		d_damping_zero_fid_readout_calibrate = np.genfromtxt(save_dir+'d_damping_zero_fid_readout_calibrate.csv')
		
		damping_zero_energy_readout = np.genfromtxt(save_dir+'damping_zero_energy_readout.csv')
		d_damping_zero_energy_readout = np.genfromtxt(save_dir+'d_damping_zero_energy_readout.csv')
		damping_zero_energy_raw = np.genfromtxt(save_dir+'damping_zero_energy_raw.csv')
		d_damping_zero_energy_raw = np.genfromtxt(save_dir+'d_damping_zero_energy_raw.csv')
		damping_zero_energy_readout_calibrate = np.genfromtxt(save_dir+'damping_zero_energy_readout_calibrate.csv')
		d_damping_zero_energy_readout_calibrate = np.genfromtxt(save_dir+'d_damping_zero_energy_readout_calibrate.csv')
	
	
	from scipy.optimize import curve_fit
	
	fit_shallow = curve_fit(exp_fit,l[l<=15],np.array(damping_raw)[l<=15],p0=[1,0.5],sigma=np.abs(d_damping_raw)[l<=15],absolute_sigma=True)
	fit_preds = np.array([ pred_from_fit(li,fit_shallow) for li in l])
	d_fit_preds = fit_preds[:,1]
	fit_preds = fit_preds[:,0]
	fit_rel_error_raw = (damping_raw - fit_preds)/fit_preds
	d_fit_rel_error_raw = np.sqrt( (d_damping_raw/fit_preds)**2 + (d_fit_preds*damping_raw/fit_preds**2)**2 )
	
	
	fit_shallow = curve_fit(exp_fit,l[l<=15],np.array(damping_readout)[l<=15],p0=[1,0.5],sigma=np.abs(d_damping_readout)[l<=15],absolute_sigma=True)
	fit_preds = np.array([ pred_from_fit(li,fit_shallow) for li in l])
	d_fit_preds = fit_preds[:,1]
	fit_preds = fit_preds[:,0]
	fit_rel_error_readout = (damping_readout - fit_preds)/fit_preds
	d_fit_rel_error_readout = np.sqrt( (d_damping_readout/fit_preds)**2 + (d_fit_preds*damping_readout/fit_preds**2)**2 )
	
	if readout_calibrate:
		fit_shallow = curve_fit(exp_fit,l[l<=15],np.array(damping_readout_calibrate)[l<=15],p0=[1,0.5],sigma=np.abs(d_damping_readout_calibrate)[l<=15],absolute_sigma=True)
		fit_preds = np.array([ pred_from_fit(li,fit_shallow) for li in l])
		d_fit_preds = fit_preds[:,1]
		fit_preds = fit_preds[:,0]
		fit_rel_error_readout_calibrate = (damping_readout_calibrate - fit_preds)/fit_preds
		d_fit_rel_error_readout_calibrate = np.sqrt( (d_damping_readout_calibrate/fit_preds)**2 + (d_fit_preds*damping_readout_calibrate/fit_preds**2)**2 )
	
	
	# plt.figure()
	
	# plt.errorbar(l,damping_readout,d_damping_readout,label='readout mitigated')
	# if readout_calibrate:
		# plt.errorbar(l,damping_readout_calibrate,d_damping_readout_calibrate,label='readout mitigated using calibration')
	# plt.errorbar(l,damping_raw,d_damping_raw,label='raw')
	# plt.xlabel('l')
	# plt.ylabel('damping')
	# plt.legend(loc='best')
	# plt.yscale('log')
	
	
	
	r0_raw = np.array(damping_raw) - 1
	dr0_raw = d_damping_raw
	score_no_mit_raw = rel_error_score(r0_raw,dr0_raw,threshold)
	
	r0_readout = np.array(damping_readout) - 1
	dr0_readout = d_damping_readout
	score_readout_only = rel_error_score(r0_readout,dr0_readout,threshold)
	
	if readout_calibrate:
		r0_readout_calibrate = np.array(damping_readout_calibrate) - 1
		dr0_readout_calibrate = d_damping_readout_calibrate
		score_readout_calibrate_only = rel_error_score(r0_readout_calibrate,dr0_readout_calibrate,threshold)
	
	[r,dr] = rel_error(damping_raw,d_damping_raw,damping_pert_raw,d_damping_pert_raw)
	score_from_pert_raw = rel_error_score(r,dr,threshold)
	[r,dr] = rel_error(damping_readout,d_damping_readout,damping_pert_readout,d_damping_pert_readout)
	score_from_pert_readout = rel_error_score(r,dr,threshold)
	if readout_calibrate:
		[r,dr] = rel_error(damping_readout_calibrate,d_damping_readout_calibrate,damping_pert_readout_calibrate,d_damping_pert_readout_calibrate)
		score_from_pert_readout_calibrate = rel_error_score(r,dr,threshold)
	
	
	[r,dr] = rel_error(damping_raw,d_damping_raw,damping_zero_fid_raw,d_damping_zero_fid_raw)
	score_zero_fid_raw = rel_error_score(r,dr,threshold)
	[r,dr] = rel_error(damping_readout,d_damping_readout,damping_zero_fid_readout,d_damping_zero_fid_readout)
	score_zero_fid_readout = rel_error_score(r,dr,threshold)
	if readout_calibrate:
		[r,dr] = rel_error(damping_readout_calibrate,d_damping_readout_calibrate,damping_zero_fid_readout_calibrate,d_damping_zero_fid_readout_calibrate)
		score_zero_fid_readout_calibrate = rel_error_score(r,dr,threshold)
		
	
	[r,dr] = rel_error(damping_raw,d_damping_raw,damping_zero_energy_raw,d_damping_zero_energy_raw)
	score_zero_energy_raw = rel_error_score(r,dr,threshold)
	[r,dr] = rel_error(damping_readout,d_damping_readout,damping_zero_energy_readout,d_damping_zero_energy_readout)
	score_zero_energy_readout = rel_error_score(r,dr,threshold)
	if readout_calibrate:
		[r,dr] = rel_error(damping_readout_calibrate,d_damping_readout_calibrate,damping_zero_energy_readout_calibrate,d_damping_zero_energy_readout_calibrate)
		score_zero_energy_readout_calibrate = rel_error_score(r,dr,threshold)
	
	
	
	score_from_small_l_raw = rel_error_score(fit_rel_error_raw,d_fit_rel_error_raw,threshold)
	score_from_small_l_readout = rel_error_score(fit_rel_error_readout,d_fit_rel_error_readout,threshold)
	if readout_calibrate:
		score_from_small_l_readout_calibrate = rel_error_score(fit_rel_error_readout_calibrate,d_fit_rel_error_readout_calibrate,threshold)
	
	
	import matplotlib.pyplot as plt
	from matplotlib import colors
	
	#cmap = colors.ListedColormap(['red','orange','yellow','green'])
	cmap = colors.ListedColormap(np.array([[255,255,204],[161,218,180],[65,182,196],[34,94,168]])/255)
	
	if readout_calibrate:
		mitigation_methods=['no mitigation','readout only','readout calibration only','pert raw','pert readout','pert readout calibration','exp_fit raw','exp_fit readout','exp_fit readout calibrate','zero fid raw','zero fid readout', 'zero fid readout calibrate','zero energy raw','zero energy readout','zero energy readout calibrate']
	else:
		mitigation_methods=['no mitigation','readout only','pert raw','pert readout','exp_fit raw','exp_fit readout','zero fid raw','zero fid readout','zero energy raw','zero energy readout']
	
	l = np.flip(l)

	score_no_mit_raw.reverse()
	score_readout_only.reverse()
	if readout_calibrate:
		score_readout_calibrate_only.reverse()
		score_from_small_l_readout_calibrate.reverse()
		score_zero_fid_readout_calibrate.reverse()
		score_from_pert_readout_calibrate.reverse()
		score_zero_energy_readout_calibrate.reverse()
	score_from_pert_raw.reverse()
	score_from_pert_readout.reverse()
	
	score_from_small_l_raw.reverse()
	score_from_small_l_readout.reverse()
	
	score_zero_fid_raw.reverse()
	score_zero_fid_readout.reverse()
	
	score_zero_energy_raw.reverse()
	score_zero_energy_readout.reverse()
	
	
	if readout_calibrate:
		effectiveness = [score_no_mit_raw,score_readout_only,score_readout_calibrate_only, score_from_pert_raw, score_from_pert_readout, score_from_pert_readout_calibrate, score_from_small_l_raw, score_from_small_l_readout, score_from_small_l_readout_calibrate,score_zero_fid_raw,score_zero_fid_readout,score_zero_fid_readout_calibrate,score_zero_energy_raw,score_zero_energy_readout,score_zero_energy_readout_calibrate]
	else:
		effectiveness = [score_no_mit_raw,score_readout_only, score_from_pert_raw, score_from_pert_readout, score_from_small_l_raw, score_from_small_l_readout,score_zero_fid_raw,score_zero_fid_readout,score_zero_energy_raw,score_zero_energy_readout]
	
	fig, ax = plt.subplots()
	im = ax.imshow(effectiveness,cmap=cmap)
	
	# Loop over data dimensions and create text annotations.
	for i in range(len(mitigation_methods)):
		for j in range(len(l)):
			if not np.isnan(effectiveness[i][j]):
				text = ax.text(j, i, effectiveness[i][j], ha="center", va="center", color="k")
				
	# We want to show all ticks...
	ax.set_xticks(np.arange(len(l)))
	ax.set_yticks(np.arange(len(mitigation_methods)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(l)
	ax.set_yticklabels(mitigation_methods)

	plt.xlabel('number of ansatz layers',fontsize = 18)
	#plt.ylabel('mitigation method',fontsize = 15)
	
	plt.title(str(n)+' qubits, '+backend_name,fontsize=15)
	fig.tight_layout()
	
	
	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=11)
	ax.tick_params(axis='x', which='minor', labelsize=11)
	
	plt.show()

