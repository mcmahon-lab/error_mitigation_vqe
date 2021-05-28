## Tools for evaluating energy expectation values on IBM quantum computers.
## Requires Qiskit
## Written by Eliott Rosenberg in 2021

import numpy as np
from qiskit import QuantumCircuit, IBMQ, execute
import re
from library import *
global account
import time

def ansatz_circuit(theta,whichPauli,measure=True,include_last_rotations=True):
	# Creates a Qiskit circuit of the ansatz used in our paper. whichPauli indicates which Pauli operator is measured in the end.
	
	n = len(whichPauli)
	l = (len(theta)-n)//n
	


	# find the backwards light cone of measured qubits
	qubitsMeasured = [i for i in range(n) if whichPauli[i] > 0]
	num_qubits_measured = len(qubitsMeasured)
	connected_qubits = light_cone(l,qubitsMeasured,n)


	# qc will be our circuit.
	# if measure:
		# qc = QuantumCircuit(n,num_qubits_measured)
	# else:
		# qc = QuantumCircuit(n)
		
	qc = QuantumCircuit(n,num_qubits_measured)
		

	# now add the layers of unitaries. Only include if in the backwards light cone of the measured operator
	for i in range(l):
		odd = i%2
		for j in range(n//2):
			q1 = 2*j+odd
			q2 = (2*j+odd+1)%n
			if q1 in connected_qubits[i]:
				qc.ry(theta[n*i+q1],q1)
				qc.ry(theta[n*i+q2],q2)
				if not odd:
					qc.cx(q1,q2)
				elif odd:
					qc.cx(q2,q1)
	
	# now add a row of 1-qubit unitaries. Absorb the measurement change of basis into these unitaries. Then measure the qubits.
	if include_last_rotations:
		for i in range(num_qubits_measured):
			qm = qubitsMeasured[i]
			if whichPauli[qm] == 1:
				# H*ry(th) = rz(pi)*sx*rz(3*pi/2 - th)*sx*rz(-pi)
				qc.rz(-np.pi,qm)
				qc.sx(qm)
				qc.rz(3*np.pi/2 - theta[n*l + qm],qm)
				qc.sx(qm)
				qc.rz(np.pi,qm)
				
			elif whichPauli[qm] == 2:
				# h*sdg*ry(th) = rz(pi/2 + th)*sx
				qc.sx(qm)
				qc.rz(np.pi/2 + theta[n*l + qm],qm)
			elif whichPauli[qm] == 3:
				qc.ry(theta[n*l + qm],qm)
		
			if measure:
				qc.measure(qm,i)
			
	return qc


def cycle_QuantumCircuit(qc, config):
	# cyclically permutes the qubits in the QuantumCircuit qc.
	# config should be an integer between -(n-1) and n.
	n = qc.num_qubits
	n_meas = qc.num_clbits
	qc_instruction = qc.to_instruction()
	qubits0 = list(range(n))
	qubits = cycle_list(qubits0,config)
	qc_rotated = QuantumCircuit(n,n_meas)
	qc_rotated.append(qc_instruction,qubits,range(n_meas))
	return qc_rotated


def load_qubit_map(machine,n):
  if machine == 'ibmq_montreal' or machine=='ibmq_toronto' or machine == 'ibmq_sydney':
    if n == 12:
        qubits = [1,4,7,10,12,13,14,11,8,5,3,2];
        #qubits = [12,15,18,21,23,24,25,22,19,16,14,13]
    elif n == 20:
        qubits = [1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2];
  elif  machine=='ibmq_rochester':
      if n == 12:
          qubits = [21,22,23,24,25,29,36,35,34,33,32,28];
  elif machine=='ibmq_cambridge':
      if n == 12:
          qubits = [0,1,2,3,4,6,13,12,11,10,9,5];
          #qubits = [7,8,9,10,11,17,23,22,21,20,19,16];
          #qubits = [11,12,13,14,15,18,27,26,25,24,23,17];
      elif n == 20:
          qubits = [0,1,2,3,4,6,13,12,11,17,23,22,21,20,19,16,7,8,9,5];
          #qubits = [0,1,2,3,4,6,13,14,15,18,27,26,25,24,23,17,11,10,9,5];
          #qubits = [7,8,9,10,11,12,13,14,15,18,27,26,25,24,23,22,21,20,19,16];
      elif n == 24:
          qubits = [0,1,2,3,4,6,13,14,15,18,27,26,25,24,23,22,21,20,19,16,7,8,9,5];
  elif machine=='ibmq_16_melbourne':
      if n == 4:
          qubits = [2,3,11,12]; # 4 qubits
      elif n == 6:
          qubits = [0,1,2,12,13,14]; # 6 qubits
      elif n == 8:
          qubits = [0,1,2,3,11,12,13,14]; # 8 qubits
      elif n == 10:
          qubits = [0,1,2,3,4,10,11,12,13,14]; # 10 qubits
      elif n == 12:
          qubits = [0,1,2,3,4,5,9,10,11,12,13,14]; # 12 qubits
  elif machine == 'ibmq_manhattan':
      if n == 12:
          qubits = [4,5,6,7,8,12,21,20,19,18,17,11]
      if n == 20:
          qubits = [0,1,2,3,4,5,6,7,8,12,21,20,19,18,17,16,15,14,13,10]
      elif n == 44:
          qubits = [0,1,2,3,4,5,6,7,8,12,21,22,23,26,37,36,35,40,49,50,51,54,64,63,62,61,60,59,58,57,56,52,43,42,41,38,27,28,29,24,15,14,13,10]
      elif n == 52:
          qubits = [0,1,2,3,4,5,6,7,8,12,21,22,23,26,37,36,35,34,33,32,31,39,45,46,47,48,49,50,51,54,64,63,62,61,60,59,58,57,56,52,43,42,41,38,27,28,29,24,15,14,13,10]
  else:
      qubits = np.arange(n)

  return qubits


def read_from_tags(varName,tags):
	from numpy import array
	for tag in tags:
		sr = re.match(varName+' = ',tag)
		if sr != None:
			# fix issue with qubits_measured_all. Change the set to a list.
			if varName == 'qubits_measured_all':
				tag = tag[:22]+'['+tag[23:-1]+']'
			exec(tag)
			return eval(varName)
			break


def bit_parity(bit_string):
	bitParity = False
	for bit in bit_string:
		if bit == '1':
			bitParity = not bitParity
	return bitParity


def P_from_counts(counts):
	# computes the expectation value of a Pauli operator
	# counts should be the counts from a single circuit, not a list of counts
	n1 = 0
	shots = 0
	for outcome_str in counts:
		if bit_parity(outcome_str):
			n1 += counts.get(outcome_str)
		shots += counts.get(outcome_str)
	
	p1 = n1/shots
	dp1 = np.sqrt( p1*(1-p1)/shots )
	
	P = 1- 2*p1
	dP = 2*dp1
	
	return P, dP


def readout_error_correct(P,dP,em,e1_minus_e0=0,de0=0,de1=0):
	# em should be the average readout error of the measured qubits (not all of the qubits)
	single_qubit = (not hasattr(em,'__iter__')) or (len(em) == 1)
	if single_qubit:
		if hasattr(em,'__iter__'):
			em = em[0]
		if hasattr(e1_minus_e0,'__iter__'):
			e1_minus_e0 = e1_minus_e0[0]
		if hasattr(de0,'__iter__'):
			de0 = de0[0]
		if hasattr(de1,'__iter__'):
			de1 = de1[0]
		P_mit = (P - e1_minus_e0)/(1-2*em)
		dP_mit = np.sqrt( (dP/(1-2*em))**2 + ( 1/(1-2*em) + (P - e1_minus_e0)/(1-2*em)**2)**2 * de0**2 + ( - 1/(1-2*em) + (P - e1_minus_e0)/(1-2*em)**2)**2 * de1**2 )
		# dP_mit for different terms are correlated because of the de0 and de1. It would be better to not treat them as independent...
		
		
	elif not single_qubit:
		em = np.array(em)
		C = np.prod(1-2*em)
		if len(em) == 2:
			dC = np.sqrt( (de0[0] * (1-2*em[1]))**2 + (de1[0] * (1-2*em[1]))**2 + (de0[1] * (1-2*em[0]))**2 + (de1[1] * (1-2*em[0]))**2 )
			# dP_mit for different terms are correlated because of the de0 and de1. It would be better to not treat them as independent...
		P_mit = P/C
		if len(em) == 2:
			dP_mit = np.sqrt( (dP/C)**2 + (P*dC/C**2)**2 )
		else:
			dP_mit = dP/C
		
		
		
	return P_mit, dP_mit
	
	
def readout_error_correct_advanced(Minv,counts,reverse_order,dMinv):
	if not reverse_order:
		counts_vector = [counts.get(bitstr,0) for bitstr in ['00','10','01','11'] ]
	else:
		counts_vector = [counts.get(bitstr,0) for bitstr in ['00','01','10','11'] ]
	counts_vector = np.array(counts_vector)
	
	shots = np.sum(counts_vector)
	
	P = 1 - 2*( counts_vector[1] + counts_vector[2])/shots
	
	
	d_counts_vector = np.sqrt(counts_vector * (shots - counts_vector)/shots)
	counts_mitigated = Minv@counts_vector
	
	d_counts_mitigated = np.sqrt( (dMinv**2)@(counts_vector**2) + Minv**2 @ d_counts_vector**2 )
	p = (counts_mitigated[1] + counts_mitigated[2])/shots
	dp = np.sqrt((d_counts_mitigated[1]**2 + d_counts_mitigated[2]**2))/shots
	P_mit = 1-2*p
	dP_mit = 2*dp
	
	
	return P_mit, dP_mit
	


def all_ising_Paulis(n):
	whichPauli_all = []
	for i in range(n):
		whichPauli = [0 for i in range(n)]
		whichPauli[i] = 3
		whichPauli[(i+1)%n] = 3
		whichPauli_all.append(whichPauli)
	for i in range(n):
		whichPauli = [0 for i in range(n)]
		whichPauli[i] = 1
		whichPauli_all.append(whichPauli)
	for i in range(n):
		whichPauli = [0 for i in range(n)]
		whichPauli[i] = 3
		whichPauli_all.append(whichPauli)
	return whichPauli_all


def all_ising_Paulis_symm(n):
	# assumes that the ansatz has cyclic permutation symmetry imposed, so we don't need to measure all of the terms.
	whichPauli_all = []
	for i in range(2):
		whichPauli = [0 for i in range(n)]
		whichPauli[i] = 3
		whichPauli[(i+1)%n] = 3
		whichPauli_all.append(whichPauli)
	for i in range(2):
		whichPauli = [0 for i in range(n)]
		whichPauli[i] = 1
		whichPauli_all.append(whichPauli)
	for i in range(2):
		whichPauli = [0 for i in range(n)]
		whichPauli[i] = 3
		whichPauli_all.append(whichPauli)
	return whichPauli_all


### reading jobs:

def energy_from_counts(counts,coeffs):
	# computes the energy from counts without any readout error mitigation
	# not set up for multiple energy evaluations per job
	E = 0
	dE2 = 0
	num_configs = len(counts)//len(coeffs)
	for term in range(len(coeffs)):
		for config in range(num_configs):
			P,dP = P_from_counts(counts[term*num_configs + config])
			E += coeffs[term] * P /num_configs
			dE2 += (coeffs[term] * dP /num_configs)**2
	return E, np.sqrt(dE2)


def energy_from_job(job,coeffs,readout_mitigate=True,calibration_job=[]):

	counts = job.result().get_counts()
	tags = job.tags()
	whichPauli_all = read_from_tags('whichPauli',tags)
	n = read_from_tags('n',tags)
	configs = read_from_tags('configs',tags)
	num_configs = len(configs[0])
	num_thetas = len(counts)//(num_configs*len(whichPauli_all))
	num_terms = len(whichPauli_all)
	
	multi_coeffs = len(np.shape(coeffs)) == 2
	if multi_coeffs:
		coeffs_all = coeffs
	
	if readout_mitigate:
		backend_name = job.backend().name()
		qubits = load_qubit_map(backend_name,n)
		properties = job.properties()
		e0 = np.array([properties.qubits[q][6].value for q in qubits])
		e1 = np.array([properties.qubits[q][5].value for q in qubits])
		
		de0 = np.sqrt(e0*(1-e0)/5000)
		de1 = np.sqrt(e1*(1-e1)/5000)
		
		e0_dates = [properties.qubits[q][6].date for q in qubits]
		e1_dates = [properties.qubits[q][5].date for q in qubits]
		em_dates = e0_dates + e1_dates
		early_date = min(em_dates)
		late_date = max(em_dates)
		job_run_date = job.time_per_step()['RUNNING']
		print('delay between readout and calibration and run is between '+str(job_run_date - early_date)+' and '+str(job_run_date - late_date))
		
		em = (e0+e1)/2
		dem = np.sqrt(de0**2 + de1**2)/2
		e1_minus_e0 = e1 - e0
		
	if calibration_job != []:
		from error_mitigation import analyze_readout_calibration_advanced
		qubits_measured_all = list(read_from_tags('qubits_measured_all',calibration_job.tags()))
		
		
		
		qubits_measured_1 = [q for q in qubits_measured_all if len(q) == 1]
		qubits_measured_2 = [q for q in qubits_measured_all if len(q) == 2]
		
		includes_one_qubit = len(qubits_measured_1) > 0
		
		# # the following treats 00 and 11 as the same outcome and 10 and 01 as the same outcome. It may not be as effective as keeping them distinct.
		# e0_2, e1_2 = analyze_readout_calibration(calibration_job)
		# em_2 = (e0_2 + e1_2)/2
		# e1_minus_e0_2 = e1_2 - e0_2
		
		e_1qubit, Minv, de_1qubit, dMinv = analyze_readout_calibration_advanced(calibration_job)
		
	
	
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
				
				counts_i = counts[which_theta*num_configs*num_terms + num_configs*term + which_config]
				P,dP = P_from_counts(counts_i)
				
				if readout_mitigate:
					if calibration_job==[] or ( len(qubits_measured) == 1 and not includes_one_qubit):
						P,dP = readout_error_correct(P,dP,em[qubits_measured_config],e1_minus_e0[qubits_measured_config], de0[qubits_measured_config],de1[qubits_measured_config] )
						
					elif len(qubits_measured) == 2:
						which_index = qubits_measured_2.index( frozenset(qubits_measured_config) )
						reversed = not (qubits_measured_config[0] == list(qubits_measured_2[which_index])[0])
						
						P, dP = readout_error_correct_advanced(Minv[which_index],counts_i,reversed,dMinv[which_index])
					elif len(qubits_measured ) == 1:
						which_index = qubits_measured_1.index( frozenset(qubits_measured_config) )
						
						P,dP = readout_error_correct(P,dP,np.mean(e_1qubit[which_index]),e_1qubit[which_index][1] - e_1qubit[which_index][0], de_1qubit[which_index][0],de_1qubit[which_index][1] )
			
				E += coeffs[term] * P /num_configs
				dE2 += (coeffs[term] * dP /num_configs )**2
		E_all.append(E)
		dE_all.append(np.sqrt(dE2))
		
	
	if num_thetas > 1:
		return E_all, dE_all
	elif num_thetas == 1:
		return E_all[0], dE_all[0]


def ising_energy_from_job(job,readout_mitigate=True,calibration_job=[]):
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
		
	return energy_from_job(job,coeffs,readout_mitigate,calibration_job)


#### submitting jobs:
	
def submit_circuits(theta,whichPauli_all,backend_name,tags=[],shots=1024,configs_all_terms=[]):
	# theta can be a list or numpy array of multiple points in parameter space.
	# if configs_all_terms is not specified, picks automatically.
	global account
	if 'account' not in globals():
		account = IBMQ.load_account()
	backend = account.get_backend(backend_name)
	
	
	multi_theta = len(np.shape(theta)) > 1
	if not multi_theta:
		theta = [theta]
	
	n = len(whichPauli_all[0]) # number of qubits
	l = len(theta[0])//n - 1  # the number of ansatz layers; depends on the ansatz
	
	
	## pick configs if not supplied
	if len(configs_all_terms) == 0:
		# load error rates
		faulty = True
		while faulty:
			qubits = load_qubit_map(backend_name,n)
			properties = backend.properties(refresh=True)
			e_cx = [properties.gate_error('cx',[qubits[i],qubits[(i+1)%n]]) for i in range(n)]
			e_sx = [properties.gate_error('sx',q) for q in qubits]
			em = [properties.readout_error(q) for q in qubits]
			faulty = max(e_cx) >= 1 or max(e_sx) >= 1 or max(em) >= 0.5
			if faulty:
				print('faulty qubits or gates. Retrying in 2 minutes')
				time.sleep(120)
		# done loading error rates
		configs_all_terms = [pick_config(l,whichPauli,e_cx,e_sx,em,minNumConfigs=4,method='largest_slopes',cutoff=0) for whichPauli in whichPauli_all]
	# done picking configs
		
		
	qc_all = []
	for th_i in theta:
		for term in range(len(whichPauli_all)):
			whichPauli = whichPauli_all[term]
			configs_term = configs_all_terms[term]
			qc = ansatz_circuit(th_i,whichPauli,True)
			for config in configs_term:
				qc_all.append( cycle_QuantumCircuit(qc,config))
				
	
	
	tags += ['n = '+str(n),'l = '+str(l),'theta = '+str(theta),'configs = '+str(configs_all_terms),'whichPauli = '+str(whichPauli_all)]
	
	for _ in range(20):
		try:
			job =  execute(qc_all, backend=backend, shots=shots, initial_layout=load_qubit_map(backend_name,n), job_tags=tags)
			break
		except:
			print('Error submitting job. Retrying.')
			time.sleep(60)
			IBMQ.load_account()
			continue
	return job


def submit_ising(n,theta,backend_name,tags=[],shots=1024,hx=1.5,hz=0.1,E=[],configs_all_terms=[]):
	return submit_circuits(theta,all_ising_Paulis(n),backend_name,tags=tags+['Ising','hx = '+str(hx),'hz = '+str(hz),'E = '+str(E)],shots=shots,configs_all_terms=configs_all_terms)


def submit_ising_symm(n,theta,backend_name,tags=[],shots=1024,hx=1.5,hz=0.1,E=[],configs_all_terms=[],input_condensed_theta=True):
	# assumes that the ansatz has permutation symmetry imposed
	
	if input_condensed_theta:
		l = len(theta[0])//2 - 1
		theta_full = [[ theta_i[theta_ALAy_to_symm(which_theta,n)] for which_theta in range(n*(l+1))] for theta_i in theta]
		theta = theta_full
	
	return submit_circuits(theta,all_ising_Paulis_symm(n),backend_name,tags=tags+['Ising','symm','hx = '+str(hx),'hz = '+str(hz),'E = '+str(E)],shots=shots,configs_all_terms=configs_all_terms)


def pick_config(l,whichPauli,e_cx,e_sx,em,minNumConfigs=4,method='largest_slopes',cutoff=0):
	n = len(whichPauli)
	configs = np.arange(-n,n)
	slopes = [damping_from_fidelities(l,whichPauli, e_cx, e_sx,config,em) for config in configs]
	
	print('Predicted slopes:')
	print(slopes)
	
	if method == 'largest_slopes':
		index = np.argmax(slopes)
		config = [configs[index]]
		print('picking config = '+str(configs[index])+' with predicted slope '+str(slopes[index]))
		
		# if predicted slope is zero, wait and try again.
		if slopes[index] == 0:
			print('Backend is not currently operational.')
			return False
		
		
		if len(config) < minNumConfigs:
			print('adding '+str(minNumConfigs-len(config))+' configs')
			indices = np.flip(np.argsort(slopes))
			config = [ configs[indices[i]] for i in range(minNumConfigs) ]
			
	elif method == 'random_with_cutoff':
		print('picking random configs with slopes > '+str(cutoff))
		indices_possible = [i for i in range(2*n) if slopes[i] > cutoff]
		if len(indices_possible) < minNumConfigs:
			print('Not enough good slopes. Retrying with cutoff -> 0.9*cutoff.')
			time.sleep(10)
			return pick_config(l,whichPauli,e_cx,e_sx,em,minNumConfigs,method,0.9*cutoff)
		rng = np.random.default_rng()
		rng.shuffle(indices_possible)
		indices = indices_possible[0:minNumConfigs]
		config = configs[indices].tolist()
	
		print('picking config = '+str(config)+', with slopes '+str(np.array(slopes)[indices].tolist()))

	
	return config
	



### submit and analyze in one step:
def Ising_E_from_theta_machine(n,theta,backend_name,tags=[],shots=1024,hx=1.5,hz=0.1,E=[],configs_all_terms=[],symm=False,readout_mitigate=True,input_condensed_theta=True):
	if symm:
		job = submit_ising_symm(n,theta,backend_name,tags,shots,hx,hz,E,configs_all_terms,input_condensed_theta)
	else:
		job = submit_ising(n,theta,backend_name,tags,shots,hx,hz,E,configs_all_terms)
	return ising_energy_from_job(job,readout_mitigate)