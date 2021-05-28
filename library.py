import numpy as np
from copy import deepcopy
import copy

def light_cone(l,qubits_measured,N):
	# returns all of the qubits in each layer that are in the backwards light cone of qubit qubit_measured. l is the total number of layers. N is number of qubits
	
	if not hasattr(qubits_measured,'__iter__'):
		qubits_measured = [qubits_measured]
	
	connected_qubits = [set(qubits_measured)]

	for layer_from_top in range(l):
		evenLayer = (l-layer_from_top)%2
		connected_qubits_this_layer = copy.deepcopy(connected_qubits[layer_from_top])
		for q in connected_qubits[layer_from_top]:
			if evenLayer == (q%2):
				qj = (q-1)%N
			else:
				qj = (q+1)%N
			connected_qubits_this_layer.add(qj)
		connected_qubits.append(connected_qubits_this_layer)
	connected_qubits.reverse()
	
	return connected_qubits


def thetas_in_light_cone_ALAy(l,qubits_measured,N):
	which_thetas = []
	lc = light_cone(l,qubits_measured,N)
	for layer in range(l+1):
		for q in lc[layer]:
			which_thetas.append(N*layer+q)
	return which_thetas


def cycle_list(v,config):
	# cyclically permutes a list
	# config should be an integer between -(n-1) and n.
	n = len(v)
	if config >= 0:
		return [v[(j+config)%n] for j in range(n)]
	elif config < 0:
		return [v[(-j+config+1)%n] for j in range(n)]


def damping_from_fidelities(l,whichPauli, e_cx, e_sx,config=0,em=[]):
	# change if using a different ansatz
	# include em if including readout errors here
	
	if config >= 0:
		e_cx = cycle_list(e_cx,config)
	elif config < 0:
		e_cx = cycle_list(e_cx,config-1)
		
	e_sx = cycle_list(e_sx,config)
	
	if len(em) != 0:
		em = cycle_list(em,config)
	
	n = len(whichPauli)
	qubitsMeasured = [i for i in range(n) if whichPauli[i] > 0]

	slope = 1.0
	
	
	# compute backwards light cone of measured qubits
	connected_qubits = light_cone(l,qubitsMeasured,n)
		
		
	# noise from ALA layers
	for layer in range(l):
		odd = layer%2
		for i in range(n//2):
			q1 = 2*i+odd
			q2 = (2*i+odd+1)%n
			if q1 in connected_qubits[layer]:
				slope = slope*(1-e_sx[q1])**2 * (1-e_sx[q2])**2 * (1-e_cx[q1])
	

	# now add error from last 1-qubit gates.
	for i in qubitsMeasured:
		if whichPauli[i] == 1 or whichPauli[i] == 3:
			slope = slope*(1-e_sx[i])**2
		elif whichPauli[i] == 2:
			slope = slope*(1-e_sx[i])
	
	
	# correcting for the difference between decoherence probability and error rate
	m = len(connected_qubits[0])
	e = 1 - slope
	p = e* 2**m/(2**m - 1)
	slope = 1-p
	
	# now add readout error if desired
	if len(em) != 0:
		for i in qubitsMeasured:
			slope = slope*(1-2*em[i])
	

	return slope


def theta_ALAy_to_symm(which_theta,n):
	which_layer = which_theta//n
	return (which_theta%2) + 2*which_layer