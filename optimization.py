import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import csv
import os
import re
import os.path
from os import path
from math import isnan
import time
from library import *
from datetime import datetime


# if you use a gpu, change this to set which gpu you use
global whichGPU
whichGPU = 3

# replace "/your_directory/saved_parameters/" with the desired directory for saved parameters

def pauli(i):
	# i = 1 for x, 2 for y, 3 for z
	if i == 1:
		P = np.array([[0,1],[1,0]])
	elif i == 2:
		P = np.array([[0,-1j],[1j,0]])
	elif i == 3:
		P = np.array([[1,0],[0,-1]])
	elif i == 0:
		P = np.eye(2)
	return P


def rot(theta,i,output_gradient=True):
	# i = 1 for x, 2 for y, 3 for z
	P = pauli(i)
	I = np.eye(2)
	U = np.cos(theta/2)*I - 1j*np.sin(theta/2)*P
	if output_gradient:
		dU = -np.sin(theta/2)/2*I - 1j*np.cos(theta/2)/2*P
	elif not output_gradient:
		dU = []
	return [U,dU]


def apply_one_qubit_matrix(U,psi, qubit, n, dU = []):
	if len(np.shape(psi)) == n and np.size(psi) == 2**n:
		first_gate = True
	elif len(np.shape(psi)) == n+1:
		first_gate = False
	elif np.size(psi) == 2**n:
		first_gate = True
		psi = np.reshape(psi,2*np.ones(n,dtype=np.int),order='F')
	output_gradient = np.size(dU) > 0

	array_of_Us = output_gradient and len(np.shape(dU)) == 3
	
	if output_gradient and first_gate:
		dpsi = multiply_state(dU,psi,qubit,array_of_Us=array_of_Us)
	elif output_gradient:
		dpsi = multiply_state(dU,psi[0],qubit,array_of_Us=array_of_Us)
	psi = multiply_state(U,psi,qubit,leave_first_index = not first_gate)
	if output_gradient and first_gate:
		psi = [psi]
	if output_gradient and array_of_Us:
		psi = np.concatenate((psi,dpsi),axis=0) # changed to concatenate.
	elif output_gradient and not array_of_Us:
		psi = np.append(psi,[dpsi],axis=0)
		
	return psi
	
	
	
def apply_one_qubit_matrix_gpu(U,psi, qubit, n, dU = [],which_theta=0):
	output_gradient = np.size(dU) > 0

	array_of_Us = output_gradient and len(np.shape(dU)) == 3

	if output_gradient:
		dpsi = multiply_state_gpu(dU,psi[0],qubit,array_of_Us=array_of_Us)
		psi[:(np.min(which_theta)+1)] = multiply_state_gpu(U,psi[:(np.min(which_theta)+1)],qubit,leave_first_index = output_gradient)
		
		psi[which_theta+1] = dpsi
	else:
		psi = multiply_state_gpu(U,psi,qubit,leave_first_index = output_gradient)

	return psi

	
def P_i(whichPauli,psi,qubit,leave_first_index=False,gpu=False):
	if not gpu:
		return multiply_state(pauli(whichPauli),psi,qubit,leave_first_index=leave_first_index)
	elif gpu:
		return multiply_state_gpu(pauli(whichPauli),psi,qubit,leave_first_index=leave_first_index)


def multiply_state(U,psi,qubits,leave_first_index=False,array_of_Us=False):

	# multiples the state psi by the matrix U. qubits indicates the indices to contract. If leave_first_index = True, then psi is treated as an array of state vectors, where the first index indices the different state vectors. All of the states in that array are multiplied by U.
	

	if not hasattr(qubits,'__iter__'):
		qubits = [qubits]
	
	n_to_contract = len(qubits)
	qubits = np.array(qubits)
	
	sh_U = np.shape(U)
	if len(sh_U) != 2*n_to_contract and not array_of_Us:
		U = np.reshape(U,2*np.ones(2*n_to_contract,dtype=np.int), order='C')

	sh = np.shape(psi)
	if not leave_first_index:
		n = int(np.log2(np.size(psi)))
	elif leave_first_index:
		n = len(sh) - 1
	if (not leave_first_index) and len(sh) != n:
		psi = np.reshape(psi,2*np.ones(n,dtype=np.int),order='F')
	
	U_axes = np.flip(np.arange(n_to_contract,2*n_to_contract)) + array_of_Us
	psi_axes = qubits + leave_first_index
	extra_indices = leave_first_index + array_of_Us
	if not array_of_Us:
		axesInv = list(np.concatenate( (np.flip(qubits)+ extra_indices, np.delete(np.arange(n+extra_indices) ,qubits+extra_indices) )))
	elif array_of_Us:
		axesInv = list(np.concatenate(  ([0], np.flip(qubits)+ extra_indices, np.delete(np.arange(1,n+extra_indices) ,qubits+extra_indices-1) )))
	axes = [axesInv.index(i) for i in range(n+extra_indices)]
	psi = np.tensordot(U,psi,axes=(U_axes,psi_axes))
	psi = np.transpose(psi,axes=axes)
	
	
	if (not leave_first_index) and len(sh) != n:
		psi = np.reshape(psi,sh,order='F')
	
	return psi
	
	
def multiply_state_gpu(U,psi,qubits,leave_first_index=False,array_of_Us=False):

	# multiples the state psi by the matrix U. qubits indicates the indices to contract. If leave_first_index = True, then psi is treated as an array of state vectors, where the first index indices the different state vectors. All of the states in that array are multiplied by U.
	
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	if not hasattr(qubits,'__iter__'):
		qubits = [qubits]
	
	n_to_contract = len(qubits)
	qubits = np.array(qubits)
	
	if type(U) == type(np.array([])):
		U = cp.array(U,dtype=cp.complex128)
	sh_U = cp.shape(U)
	if len(sh_U) != 2*n_to_contract and not array_of_Us:
		U = cp.reshape(U,2*np.ones(2*n_to_contract,dtype=np.int), order='C')
	

	sh = cp.shape(psi)
	if not leave_first_index:
		n = int(np.log2(cp.size(psi)))
	elif leave_first_index:
		n = len(sh) - 1
	if (not leave_first_index) and len(sh) != n:
		psi = cp.reshape(psi,2*np.ones(n,dtype=np.int),order='F')
	
	U_axes = np.flip(np.arange(n_to_contract,2*n_to_contract)) + array_of_Us
	psi_axes = qubits + leave_first_index
	extra_indices = leave_first_index + array_of_Us
	if not array_of_Us:
		axesInv = list(np.concatenate( (np.flip(qubits)+ extra_indices, np.delete(np.arange(n+extra_indices) ,qubits+extra_indices) )))
	elif array_of_Us:
		axesInv = list(np.concatenate(  ([0], np.flip(qubits)+ extra_indices, np.delete(np.arange(1,n+extra_indices) ,qubits+extra_indices-1) )))
	axes = [axesInv.index(i) for i in range(n+extra_indices)]
	psi = cp.tensordot(U,psi,axes=(U_axes,psi_axes))
	psi = cp.transpose(psi,axes=axes)
	
	
	if (not leave_first_index) and len(sh) != n:
		psi = cp.reshape(psi,sh,order='F')
	
	return psi


def ALAy_state(theta,n,output_gradient=True,includeH=False):
	# ansatz from Fig 4 of https://arxiv.org/pdf/2001.00550.pdf
	C01 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
	if includeH:
		H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
	l = (len(theta)-n)//n
	psi = np.zeros(2**n)
	psi[0] = 1
	#psi = np.reshape(psi,2*np.ones(n,dtype=np.int),order='F')
	
	# now act with ansatz layers.
	whichTheta = 0
	for layer in range(l):
		# first apply one qubit gates
		for i in range(n):
			[U,dU] = rot(theta[whichTheta],2)
			if not output_gradient:
				dU = []
			whichTheta += 1
			psi = apply_one_qubit_matrix(U,psi, i, n, dU = dU)
		# now apply Hs
		if includeH and layer == 0:
			for i in range(n//2):
				psi = apply_one_qubit_matrix(H,psi,2*i+1,n)
		# now apply cnots
		odd = layer%2
		for j in range(n//2):
			q1 = 2*j+odd
			q2 = (2*j+odd+1)%n
			if not odd:
				psi = multiply_state(C01,psi,[q1,q2],leave_first_index=output_gradient)
			elif odd:
				psi = multiply_state(C01,psi,[q2,q1],leave_first_index=output_gradient)
		
	# now apply Hs
	if includeH and layer == 0:
		for i in range(n//2):
			psi = apply_one_qubit_matrix(H,psi,2*i+1)
	
	# now act with last row of 1-qubit unitaries
	for i in range(n):
		[U,dU] = rot(theta[whichTheta],2)
		if not output_gradient:
			dU = []
		psi = apply_one_qubit_matrix(U,psi, i, n, dU = dU)
		whichTheta +=1
	
	return psi
	
	
def ALAy_state_lc(theta,n,qubits_measured,output_gradient=True):
	# don't include Hs, i.e. this is ALAy_cx.
	C01 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
	l = (len(theta)-n)//n
	lc = light_cone(l,qubits_measured,n)
	qubits_in_lc = list(lc[0])
	m = len(qubits_in_lc) # number of qubits that we need to represent
	psi = np.zeros(2**m)
	psi[0] = 1
	for layer in range(l):
		for q in lc[layer]:
			[U,dU] = rot(theta[n*layer+q],2)
			if not output_gradient:
				dU = []
			vq = qubits_in_lc.index(q)
			psi = apply_one_qubit_matrix(U,psi, vq, m, dU = dU)
		# now apply cnots
		odd = layer%2
		for j in range(n//2):
			q1 = 2*j+odd
			q2 = (2*j+odd+1)%n
			if q1 in lc[layer]:
				vq1 = qubits_in_lc.index(q1)
				vq2 = qubits_in_lc.index(q2)
				if not odd:
					psi = multiply_state(C01,psi,[vq1,vq2],leave_first_index=output_gradient)
				elif odd:
					psi = multiply_state(C01,psi,[vq2,vq1],leave_first_index=output_gradient)
	# now act with last row of 1-qubit unitaries
	for q in lc[l]:
		[U,dU] = rot(theta[n*l+q],2)
		if not output_gradient:
			dU = []
		vq = qubits_in_lc.index(q)
		psi = apply_one_qubit_matrix(U,psi, vq, m, dU = dU)
	
	return psi


def ALAy_state_lc_symm(theta,n,qubits_measured,output_gradient=True):
	# don't include Hs, i.e. this is ALAy_cx.
	# impose permutation symmetry.
	# not finished.
	C01 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
	l = (len(theta)-2)//2
	lc = light_cone(l,qubits_measured,n)
	qubits_in_lc = list(lc[0])
	m = len(qubits_in_lc) # number of qubits that we need to represent
	psi = np.zeros(2**m)
	psi[0] = 1
	for layer in range(l):
		for q in lc[layer]:
			[U,dU] = rot(theta[2*layer+(q%2)],2)
			if not output_gradient:
				dU = []
			vq = qubits_in_lc.index(q)
			psi = apply_one_qubit_matrix(U,psi, vq, m, dU = dU)
		# now apply cnots
		odd = layer%2
		for j in range(n//2):
			q1 = 2*j+odd
			q2 = (2*j+odd+1)%n
			if q1 in lc[layer]:
				vq1 = qubits_in_lc.index(q1)
				vq2 = qubits_in_lc.index(q2)
				if not odd:
					psi = multiply_state(C01,psi,[vq1,vq2],leave_first_index=output_gradient)
				elif odd:
					psi = multiply_state(C01,psi,[vq2,vq1],leave_first_index=output_gradient)
	# now act with last row of 1-qubit unitaries
	for q in lc[l]:
		[U,dU] = rot(theta[n*l+q],2)
		if not output_gradient:
			dU = []
		vq = qubits_in_lc.index(q)
		psi = apply_one_qubit_matrix(U,psi, vq, m, dU = dU)
	
	return psi



def ALAy_state_lc_gpu(theta,n,qubits_measured,output_gradient=True):
	# don't include Hs, i.e. this is ALAy_cx.
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	C01 = cp.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],dtype=cp.int8)
	l = (len(theta)-n)//n
	lc = light_cone(l,qubits_measured,n)
	qubits_in_lc = list(lc[0])
	
	thetas_in_lc = 0
	for q_l in lc:
		thetas_in_lc += len(q_l) # number of thetas in light cone
	
	m = len(qubits_in_lc) # number of qubits that we need to represent
	
	shape = [2 for _ in range(m)]
	if output_gradient:
		shape = [thetas_in_lc + 1]+shape
	psi = cp.zeros(shape,dtype = cp.complex128)
	if output_gradient:
		psi[tuple([0 for _ in range(m+1)])] = 1
	else:
		psi[tuple([0 for _ in range(m)])] = 1
	
	which_theta = 0
	for layer in range(l):
		for q in lc[layer]:
			[U,dU] = rot(theta[n*layer+q],2)
			if not output_gradient:
				dU = []
			vq = qubits_in_lc.index(q)
			psi = apply_one_qubit_matrix_gpu(U,psi, vq, m, dU = dU,which_theta = which_theta)
			which_theta += 1
		# now apply cnots
		odd = layer%2
		for j in range(n//2):
			q1 = 2*j+odd
			q2 = (2*j+odd+1)%n
			if q1 in lc[layer]:
				vq1 = qubits_in_lc.index(q1)
				vq2 = qubits_in_lc.index(q2)
				if not odd:
					psi = multiply_state_gpu(C01,psi,[vq1,vq2],leave_first_index=output_gradient)
				elif odd:
					psi = multiply_state_gpu(C01,psi,[vq2,vq1],leave_first_index=output_gradient)
	# now act with last row of 1-qubit unitaries
	for q in lc[l]:
		[U,dU] = rot(theta[n*l+q],2)
		if not output_gradient:
			dU = []
		vq = qubits_in_lc.index(q)
		psi = apply_one_qubit_matrix_gpu(U,psi, vq, m, dU = dU, which_theta = which_theta)
		which_theta += 1
	
	return psi


def ALAy_state_lc_gpu_symm(theta,n,qubits_measured,output_gradient=True):
	# don't include Hs, i.e. this is ALAy_cx.
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	C01 = cp.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],dtype=cp.int8)
	l = (len(theta)-2)//2
	lc = light_cone(l,qubits_measured,n)
	qubits_in_lc = list(lc[0])
	
	thetas_in_lc = 0  # number of thetas in light cone
	for q_l in lc:
		if len(q_l) > 1:
			thetas_in_lc += 2 
		elif len(q_l) == 1:
			thetas_in_lc += 1
	
	m = len(qubits_in_lc) # number of qubits that we need to represent
	
	shape = [2 for _ in range(m)]
	if output_gradient:
		shape = [thetas_in_lc + 1]+shape
	psi = cp.zeros(shape,dtype = cp.complex128)
	if output_gradient:
		psi[tuple([0 for _ in range(m+1)])] = 1
	else:
		psi[tuple([0 for _ in range(m)])] = 1
	
	which_theta = 0
	for layer in range(l):
		lc_layer_even = [i for i in lc[layer] if (i%2)==0]
		lc_layer_odd = [i for i in lc[layer] if (i%2) == 1]
		for q in lc_layer_even:
			[U,dU] = rot(theta[2*layer+(q%2)],2)
			if not output_gradient:
				dU = []
			vq = qubits_in_lc.index(q)
			psi = apply_one_qubit_matrix_gpu(U,psi, vq, m, dU = dU,which_theta = which_theta)
		which_theta += 1
		for q in lc_layer_odd:
			[U,dU] = rot(theta[2*layer+(q%2)],2)
			if not output_gradient:
				dU = []
			vq = qubits_in_lc.index(q)
			psi = apply_one_qubit_matrix_gpu(U,psi, vq, m, dU = dU,which_theta = which_theta)
		which_theta += 1
		# now apply cnots
		odd = layer%2
		for j in range(n//2):
			q1 = 2*j+odd
			q2 = (2*j+odd+1)%n
			if q1 in lc[layer]:
				vq1 = qubits_in_lc.index(q1)
				vq2 = qubits_in_lc.index(q2)
				if not odd:
					psi = multiply_state_gpu(C01,psi,[vq1,vq2],leave_first_index=output_gradient)
				elif odd:
					psi = multiply_state_gpu(C01,psi,[vq2,vq1],leave_first_index=output_gradient)
	# now act with last row of 1-qubit unitaries
	lc_l_even = [q for q in lc[l] if q%2 == 0]
	lc_l_odd = [q for q in lc[l] if q%2 == 1]
	for q in lc_l_even:
		[U,dU] = rot(theta[2*l+q%2],2)
		if not output_gradient:
			dU = []
		vq = qubits_in_lc.index(q)
		psi = apply_one_qubit_matrix_gpu(U,psi, vq, m, dU = dU, which_theta = which_theta)
	if len(lc_l_even) > 0:
		which_theta += 1
	for q in lc_l_odd:
		[U,dU] = rot(theta[2*l+q%2],2)
		if not output_gradient:
			dU = []
		vq = qubits_in_lc.index(q)
		psi = apply_one_qubit_matrix_gpu(U,psi, vq, m, dU = dU, which_theta = which_theta)
	
	return psi




def E_from_theta(theta,paulis_in_H,coeffs,output_gradient=True,directory='ising',saveOutput=True):
	# assumes ansatz is ALAy_cx
	# not tested yet.
	n = len(paulis_in_H[0])
	l = (len(theta)-n)//n
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
	for i in range(len(coeffs)):
		Phat = paulis_in_H[i]
		qubits_measured = [j for j in range(n) if Phat[j]>0]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc(theta,n,qubits_measured,output_gradient)
		ket = deepcopy(psi)
		for j in range(len(qubits_measured)):
			vq = qubits_in_lc.index(qubits_measured[j])
			ket = P_i(Phat[qubits_measured[j]],ket,vq,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += coeffs[i]*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += coeffs[i]*P
	
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory == 'ising':
			directory = '/your_directory/saved_parameters/ising/ALAy_cx/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
		else:
			directory = '/your_directory/saved_parameters/'+directory
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E


def ising_E_from_theta_lc(theta,n,hx,hz,output_gradient=False,saveOutput=True,directory='default'):
	# ansatz is ALAy_cx
	l = (len(theta)-n)//n
	
	
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
	
	for i in range(n):
		qubits_measured = [i,(i+1)%n]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc(theta,n,qubits_measured,output_gradient)
		vq = qubits_in_lc.index(i)
		ket = P_i(3,psi,vq,leave_first_index=output_gradient)
		
		# Z:
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= hz*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E -= hz*P
		
		# ZZ:
		vq2 = qubits_in_lc.index( (i+1)%n)
		ket = P_i(3,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E -= P
		
		# X:
		vq = qubits_in_lc.index(i)
		ket = P_i(1,psi,vq,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= hx*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E -= hx*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory=='default':
			directory = '/your_directory/saved_parameters/ising/ALAy_cx/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E
		


def ising_E_from_theta(theta,n,hx,hz,output_gradient=False,directory='default',ansatz='ALAy_cx',includeH=False,saveOutput=True):
	if ansatz == 'ALAy_cx':
		ansatz = 'ALAy'
		includeH = False
	if directory == 'default':
		if ansatz=='PSA':
			l = (len(theta)-3*n)//(9*n//2)
			directory = '/your_directory/saved_parameters/ising/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
		elif ansatz=='ALAy' and includeH:
			l = (len(theta)-n)//n
			directory = '/your_directory/saved_parameters/ising/ALAy/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
		elif ansatz=='ALAy' and not includeH:
			l = (len(theta)-n)//n
			directory = '/your_directory/saved_parameters/ising/ALAy_cx/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
	if ansatz == 'PSA':
		psi = ALA_state(theta,n,output_gradient=output_gradient)
	elif ansatz == 'ALAy':
		psi = ALAy_state(theta,n,output_gradient=output_gradient,includeH=includeH)
	E = ising_H(hx,hz,psi,output_gradient=output_gradient)
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E




def ising_E_from_theta_gpu(theta,n,hx,hz,output_gradient=False,saveOutput=True,directory='default'):
	# ansatz is ALAy_cx
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	l = (len(theta)-n)//n
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
		
	psi = ALAy_state_lc_gpu(theta,n,range(n),output_gradient)
		
	for i in range(n):
	
		ket = P_i(3,psi,i,leave_first_index=output_gradient,gpu=True)
		
		# Z:
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(n),range(1,n+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,range(n),n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= hz*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= hz*P
		
		# ZZ:
		ket = P_i(3,ket,(i+1)%n,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(n),range(1,n+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,range(n),n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= P
		
		# X:
		ket = P_i(1,psi,i,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(n),range(1,n+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,range(n),n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= hx*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= hx*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory == 'default':
			directory = '/your_directory/saved_parameters/ising/ALAy_cx/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E




def ising_E_from_theta_lc_gpu(theta,n,hx,hz,output_gradient=False,saveOutput=True,directory='default'):
	# ansatz is ALAy_cx
	
	l = (len(theta)-n)//n
	if 2*l > n and not output_gradient:
		# more efficient to use full state vector:
		return ising_E_from_theta_gpu(theta,n,hx,hz,output_gradient,saveOutput,directory)
	
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
		
	for i in range(n):
		qubits_measured = [i,(i+1)%n]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc_gpu(theta,n,qubits_measured,output_gradient)
		
		vq = qubits_in_lc.index(i)
		ket = P_i(3,psi,vq,leave_first_index=output_gradient,gpu=True)
		
		# Z:
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= hz*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= hz*P
		
		# ZZ:
		vq2 = qubits_in_lc.index( (i+1)%n)
		ket = P_i(3,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= P
		
		# X:
		vq = qubits_in_lc.index(i)
		ket = P_i(1,psi,vq,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= hx*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= hx*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory == 'default':
			directory = '/your_directory/saved_parameters/ising/ALAy_cx/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E
		


def ising_E_from_theta_lc_symm(theta,n,hx,hz,output_gradient=False,gpu=False,saveOutput=True):
	from library import theta_ALAy_to_symm
	l = (len(theta)-2)//2
	theta_full = [ theta[theta_ALAy_to_symm(which_theta,n)] for which_theta in range(n*(l+1))]
	directory='/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)
	if gpu:
		E = ising_E_from_theta_lc_gpu(theta_full,n,hx,hz,output_gradient=output_gradient,saveOutput=saveOutput,directory=directory)
	elif not gpu:
		E = ising_E_from_theta_lc(theta_full,n,hx,hz,output_gradient=output_gradient,saveOutput=saveOutput,directory=directory)
	if output_gradient:
		dE_full = E[1]
		dE = np.zeros( 2*(l+1) )
		for i in range(	n*(l+1)):
			dE[ theta_ALAy_to_symm(i,n) ] += dE_full[i]
		E = (E[0], dE)
	return E



def heisenberg_E_from_theta_lc_symm(theta,n,J,B,output_gradient=False,gpu=True,saveOutput=True):
	from library import theta_ALAy_to_symm
	l = (len(theta)-2)//2
	theta_full = [ theta[theta_ALAy_to_symm(which_theta,n)] for which_theta in range(n*(l+1))]
	directory='/your_directory/saved_parameters/Heisenberg/ALAy_symm/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)
	if gpu:
		E = Heisenberg_E_from_theta_lc_gpu(theta_full,n,J,B,output_gradient=output_gradient,saveOutput=saveOutput,directory=directory)
	elif not gpu:
		E = Heisenberg_E_from_theta_lc(theta_full,n,J,B,output_gradient=output_gradient,saveOutput=saveOutput,directory=directory)
	if output_gradient:
		dE_full = E[1]
		dE = np.zeros( 2*(l+1) )
		for i in range(n*(l+1)):
			dE[ theta_ALAy_to_symm(i,n) ] += dE_full[i]
		E = (E[0], dE)
	return E


def Heisenberg_E_from_theta_lc(theta,n,J,B,output_gradient=False,saveOutput=True,directory='default'):
	# ansatz is ALAy_cx.
	l = (len(theta)-n)//n
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
		
	for i in range(n):
		qubits_measured = [i,(i+1)%n]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc(theta,n,qubits_measured,output_gradient)
		
		vq = qubits_in_lc.index(i)
		ket = P_i(3,psi,vq,leave_first_index=output_gradient)
		
		# Z:
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += B*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += B*P
		
		# ZZ:
		vq2 = qubits_in_lc.index( (i+1)%n)
		ket = P_i(3,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += J*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += J*P
		
		# XX:
		ket = P_i(1,psi,vq,leave_first_index=output_gradient)
		ket = P_i(1,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += J*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += J*P
			
		# YY:
		ket = P_i(2,psi,vq,leave_first_index=output_gradient)
		ket = P_i(2,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += J*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += J*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory=='default':
			directory = '/your_directory/saved_parameters/Heisenberg/ALAy_cx/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E
		




def Heisenberg_E_from_theta_lc_gpu(theta,n,J,B,output_gradient=False,saveOutput=True,directory='default'):
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	l = (len(theta)-n)//n
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
		
	for i in range(n):
		qubits_measured = [i,(i+1)%n]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc_gpu(theta,n,qubits_measured,output_gradient)
		
		vq = qubits_in_lc.index(i)
		ket = P_i(3,psi,vq,leave_first_index=output_gradient,gpu=True)
		
		# Z:
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += B*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E += B*P
		
		# ZZ:
		vq2 = qubits_in_lc.index( (i+1)%n)
		ket = P_i(3,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += J*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E += J*P
		
		# XX:
		ket = P_i(1,psi,vq,leave_first_index=output_gradient,gpu=True)
		ket = P_i(1,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += J*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E += J*P
			
		# YY:
		ket = P_i(2,psi,vq,leave_first_index=output_gradient,gpu=True)
		ket = P_i(2,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += J*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E += J*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory=='default':
			directory = '/your_directory/saved_parameters/Heisenberg/ALAy_cx/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E
		



def XY_E_from_theta_lc(theta,n,Jz,Jxy,output_gradient=False,saveOutput=True,directory='default'):
	# ansatz is ALAy_cx.
	l = (len(theta)-n)//n
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
		
	for i in range(n):
		qubits_measured = [i,(i+1)%n]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc(theta,n,qubits_measured,output_gradient)
		
		vq = qubits_in_lc.index(i)
		ket = P_i(3,psi,vq,leave_first_index=output_gradient)
		
		# ZZ:
		vq2 = qubits_in_lc.index( (i+1)%n)
		ket = P_i(3,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += Jz*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += Jz*P
		
		# XX:
		ket = P_i(1,psi,vq,leave_first_index=output_gradient)
		ket = P_i(1,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= Jxy*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E += -Jxy*P
			
		# YY:
		ket = P_i(2,psi,vq,leave_first_index=output_gradient)
		ket = P_i(2,ket,vq2,leave_first_index=output_gradient)
		if output_gradient:
			P = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) ))
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= Jxy*P
		elif not output_gradient:
			P = np.vdot(psi,ket).real
			E -= Jxy*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory=='default':
			directory = '/your_directory/saved_parameters/XY/ALAy_cx/n'+str(n)+'_l'+str(l)+'_Jz'+str(Jz)+'_Jxy'+str(Jxy)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E




def XY_E_from_theta_lc_gpu(theta,n,Jz,Jxy,output_gradient=False,saveOutput=True,directory='default'):
	import cupy as cp
	global whichGPU
	cp.cuda.Device(whichGPU).use()
	l = (len(theta)-n)//n
	if output_gradient:
		E = np.zeros(len(theta)+1)
	elif not output_gradient:
		E = 0
		
	for i in range(n):
		qubits_measured = [i,(i+1)%n]
		lc = light_cone(l,qubits_measured,n)
		qubits_in_lc = list(lc[0])
		m = len(qubits_in_lc)
		psi = ALAy_state_lc_gpu(theta,n,qubits_measured,output_gradient)
		
		vq = qubits_in_lc.index(i)
		ket = P_i(3,psi,vq,leave_first_index=output_gradient,gpu=True)
		
		# ZZ:
		vq2 = qubits_in_lc.index( (i+1)%n)
		ket = P_i(3,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] += Jz*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E += Jz*P
		
		# XX:
		ket = P_i(1,psi,vq,leave_first_index=output_gradient,gpu=True)
		ket = P_i(1,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= Jxy*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= Jxy*P
			
		# YY:
		ket = P_i(2,psi,vq,leave_first_index=output_gradient,gpu=True)
		ket = P_i(2,ket,vq2,leave_first_index=output_gradient,gpu=True)
		if output_gradient:
			P = cp.real(cp.tensordot( cp.conj(psi[0]), ket, axes = (range(m),range(1,m+1)) )).get()
			P[1:] = 2*P[1:]
			thetas_in_lc = np.array(thetas_in_light_cone_ALAy(l,qubits_measured,n))
			E[ np.insert(thetas_in_lc+1,0,0)] -= Jxy*P
		elif not output_gradient:
			P = cp.real(cp.vdot(psi,ket)).get()
			E -= Jxy*P
					
	if output_gradient:
		dE = E[1:]
		E = E[0]
	if saveOutput:
		if directory=='default':
			directory = '/your_directory/saved_parameters/XY/ALAy_cx/n'+str(n)+'_l'+str(l)+'_Jz'+str(Jz)+'_Jxy'+str(Jxy)
		saveParams(theta,E,directory)
	print(E)
	if output_gradient:
		return (E,dE)
	else:
		return E






def ising_H(hx,hz,psi,hy=0,output_gradient=False,add_stochastic_noise=False,shots=1024):
	if not add_stochastic_noise:
		ket = np.zeros_like(psi)
		if not output_gradient:
			n = int(np.log2(np.size(psi)))
		elif output_gradient:
			n = len(np.shape(psi))-1
		for i in range(n):
			Zi = P_i(3,psi,i,leave_first_index=output_gradient)
			ket -= P_i(3,Zi,(i+1)%n,leave_first_index=output_gradient) + hz*Zi + hx*P_i(1,psi,i,leave_first_index=output_gradient)
			if hy != 0:
				ket -= hy*P_i(2,psi,i,leave_first_index=output_gradient)
		if output_gradient:
			E = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(n),range(1,n+1)) ))
			E[1:] = 2*E[1:]
			return E
		elif not output_gradient:
			return np.vdot(psi,ket).real
	elif add_stochastic_noise:
		rng = np.random.default_rng()
		E = 0
		E_exact = 0
		n = int(np.log2(np.size(psi)))
		for i in range(n):
			Zi_state = P_i(3,psi,i)
			Zi = np.vdot(psi,Zi_state).real
			result = rng.binomial( 1, (1-Zi)/2, shots)
			p1 = np.sum(result)/shots
			E -= hz * (1-2*p1)
			E_exact -= hz*Zi
			
			ZZi_state = P_i(3,Zi_state,(i+1)%n)
			ZZi = np.vdot(psi,ZZi_state).real
			result = rng.binomial( 1, (1-ZZi)/2, shots)
			p1 = np.sum(result)/shots
			E -= (1-2*p1)
			E_exact -= ZZi
			
			Xi_state = P_i(1,psi,i)
			Xi = np.vdot(psi,Xi_state).real
			result = rng.binomial( 1, (1-Xi)/2, shots)
			p1 = np.sum(result)/shots
			E -= hx * (1-2*p1)
			E_exact -= hx*Xi
		return [E,E_exact]



def heisenberg_H(J,psi,output_gradient=False,add_stochastic_noise=False,shots=1024):
	# normalizes H so that B = 1.
	if not add_stochastic_noise:
		ket = np.zeros_like(psi)
		if not output_gradient:
			n = int(np.log2(np.size(psi)))
		elif output_gradient:
			n = len(np.shape(psi))-1
		for i in range(n):
			Zi = P_i(3,psi,i,leave_first_index=output_gradient)
			ket += J*P_i(3,Zi,(i+1)%n,leave_first_index=output_gradient) + Zi

			Xi = P_i(1,psi,i,leave_first_index=output_gradient)
			ket += J*P_i(1,Xi,(i+1)%n,leave_first_index=output_gradient)
			
			Yi = P_i(2,psi,i,leave_first_index=output_gradient)
			ket += J*P_i(2,Yi,(i+1)%n,leave_first_index=output_gradient)
		if output_gradient:
			E = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(n),range(1,n+1)) ))
			E[1:] = 2*E[1:]
			return E
		elif not output_gradient:
			return np.vdot(psi,ket).real
	elif add_stochastic_noise:
		rng = np.random.default_rng()
		E = 0
		E_exact = 0
		n = int(np.log2(np.size(psi)))
		for i in range(n):
			Zi_state = P_i(3,psi,i)
			Zi = np.vdot(psi,Zi_state).real
			result = rng.binomial( 1, (1-Zi)/2, shots)
			p1 = np.sum(result)/shots
			E += (1-2*p1)
			E_exact += Zi
			
			ZZi_state = P_i(3,Zi_state,(i+1)%n)
			ZZi = np.vdot(psi,ZZi_state).real
			result = rng.binomial( 1, (1-ZZi)/2, shots)
			p1 = np.sum(result)/shots
			E += J*(1-2*p1)
			E_exact += J*ZZi
			
			
			Xi_state = P_i(1,psi,i)
			XXi_state = P_i(1,Xi_state,(i+1)%n)
			XXi = np.vdot(psi,XXi_state).real
			result = rng.binomial( 1, (1-XXi)/2, shots)
			p1 = np.sum(result)/shots
			E += J*(1-2*p1)
			E_exact += J*XXi


			Yi_state = P_i(2,psi,i)
			YYi_state = P_i(2,Yi_state,(i+1)%n)
			YYi = np.vdot(psi,YYi_state).real
			result = rng.binomial( 1, (1-YYi)/2, shots)
			p1 = np.sum(result)/shots
			E += J*(1-2*p1)
			E_exact += J*YYi
			

		return [E,E_exact]



def XY_H(Jz,Jxy,psi,output_gradient=False):
	ket = np.zeros_like(psi)
	if not output_gradient:
		n = int(np.log2(np.size(psi)))
	elif output_gradient:
		n = len(np.shape(psi))-1
	for i in range(n):
		Zi = P_i(3,psi,i,leave_first_index=output_gradient)
		ket += Jz*P_i(3,Zi,(i+1)%n,leave_first_index=output_gradient)

		Xi = P_i(1,psi,i,leave_first_index=output_gradient)
		ket -= Jxy*P_i(1,Xi,(i+1)%n,leave_first_index=output_gradient)
		
		Yi = P_i(2,psi,i,leave_first_index=output_gradient)
		ket -= Jxy*P_i(2,Yi,(i+1)%n,leave_first_index=output_gradient)
	if output_gradient:
		E = np.real(np.tensordot( np.conj(psi[0]), ket, axes = (range(n),range(1,n+1)) ))
		E[1:] = 2*E[1:]
		return E
	elif not output_gradient:
		return np.vdot(psi,ket).real
		
		
		return E



def optimize(n,l,hx=1.5,hz=0.1,method='BFGS',gpu=True,start_from_saved=False,theta0=[],jac=True):
	if start_from_saved:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_cx/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)+'/theta.csv',delimiter=',')
	elif len(theta0) == 0:
		theta0 = np.random.rand(n+l*n)*2*np.pi
	if gpu:
		return minimize(ising_E_from_theta_lc_gpu,theta0,args=(n,hx,hz,jac),method=method,jac=jac)
	elif not gpu:
		return minimize(ising_E_from_theta_lc,theta0,args=(n,hx,hz,jac),method=method,jac=jac)




def optimize_symm(n,l,hx,hz,method='BFGS',maxiter=2000,jac=False,start_from_saved=False,gpu=True,theta0=[],start_from_previous_hx=False,start_from_next_hx=False):
	if start_from_previous_hx:
		if hx == 0.3:
			theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx0.2_hz'+str(hz)+'/theta.csv')
		elif hx == 0.4:
			theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx0.3_hz'+str(hz)+'/theta.csv')
		elif hx == 0.5:
			theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx0.4_hz'+str(hz)+'/theta.csv')
		elif hx == 0.2:
			theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx0.1_hz'+str(hz)+'/theta.csv')
	elif start_from_next_hx:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx'+str(hx+0.1)+'_hz'+str(hz)+'/theta.csv')
	elif start_from_saved:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/ising/ALAy_symm/n'+str(n)+'_l'+str(l)+'_hx'+str(hx)+'_hz'+str(hz)+'/theta.csv')
	if len(theta0) > 2*(l+1):
		theta0_symm = np.zeros(2+2*l)
		from library import theta_ALAy_to_symm
		for i in range(len(theta0)):
			theta0_symm[theta_ALAy_to_symm(i,n)] = theta0[i]
		theta0 = theta0_symm
	elif len(theta0) == 0:
		theta0 = np.random.rand(2+l*2)*2*np.pi
	return minimize(ising_E_from_theta_lc_symm,theta0,args=(n,hx,hz,jac,gpu),method=method,jac=jac,options={'maxiter':maxiter})




def optimize_symm_Heisenberg(n,l,J,B,method='TNC',maxiter=2000,jac=True,start_from_saved=False,gpu=False,start_from_previous_l=False,theta0=[]):
	if start_from_saved:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/Heisenberg/ALAy_symm/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)+'/theta.csv')
	elif start_from_previous_l:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/Heisenberg/ALAy_symm/n'+str(n)+'_l'+str(l-1)+'_B'+str(B)+'_J'+str(J)+'/theta.csv')
		theta0_symm = np.zeros(2*l)
		from library import theta_ALAy_to_symm
		for i in range(len(theta0)):
			theta0_symm[theta_ALAy_to_symm(i,n)] = theta0[i]
		theta0 = theta0_symm
		theta0 = np.concatenate(([0,0],theta0))
		
	if len(theta0) > 2*(l+1):
		theta0_symm = np.zeros(2+2*l)
		from library import theta_ALAy_to_symm
		for i in range(len(theta0)):
			theta0_symm[theta_ALAy_to_symm(i,n)] = theta0[i]
		theta0 = theta0_symm
	elif len(theta0) == 0:
		theta0 = np.random.rand(2+l*2)*2*np.pi
	return minimize(heisenberg_E_from_theta_lc_symm,theta0,args=(n,J,B,jac,gpu),method=method,jac=jac,options={'maxiter':maxiter})


def optimize_Heisenberg(n,l,J,B,method='TNC',start_from_saved=False,start_from_previous_l=False,gpu=False,theta0=[],jac=True):
	if start_from_previous_l:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/Heisenberg/ALAy_cx/n'+str(n)+'_l'+str(l-1)+'_B'+str(B)+'_J'+str(J)+'/theta.csv')
		# cyclically permute thetas
		for layer in range(l+1):
			theta_l = theta0[n*l:(n*(l+1))]
			theta_l = np.roll(theta_l,-1)
			theta0[n*l:(n*(l+1))] = theta_l
		theta0 = np.concatenate( (np.zeros(n), theta0) )
	elif start_from_saved:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/Heisenberg/ALAy_cx/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)+'/theta.csv')
		#theta0 = np.genfromtxt('/your_directory/saved_parameters/Heisenberg/ALAy_symm/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)+'/theta.csv')
	elif len(theta0)==0:
		theta0 = np.random.rand(n+l*n)*2*np.pi
	if gpu:
		return minimize(Heisenberg_E_from_theta_lc_gpu,theta0,args=(n,J,B,jac),method=method,jac=jac)
	else:
		return minimize(Heisenberg_E_from_theta_lc,theta0,args=(n,J,B,jac),method=method,jac=jac)



def optimize_XY(n,l,Jz,Jxy,method='TNC',start_from_saved=False,start_from_previous_l=False,gpu=False,theta0=[],jac=True):
	if start_from_previous_l:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/XY/ALAy_cx/n'+str(n)+'_l'+str(l-1)+'_Jz'+str(Jz)+'_Jxy'+str(Jxy)+'/theta.csv')
		# cyclically permute thetas
		for layer in range(l+1):
			theta_l = theta0[n*l:(n*(l+1))]
			theta_l = np.roll(theta_l,-1)
			theta0[n*l:(n*(l+1))] = theta_l
		theta0 = np.concatenate( (np.zeros(n), theta0) )
	elif start_from_saved:
		theta0 = np.genfromtxt('/your_directory/saved_parameters/XY/ALAy_cx/n'+str(n)+'_l'+str(l)+'_Jz'+str(Jz)+'_Jxy'+str(Jxy)+'/theta.csv')
		#theta0 = np.genfromtxt('/your_directory/saved_parameters/Heisenberg/ALAy_symm/n'+str(n)+'_l'+str(l)+'_B'+str(B)+'_J'+str(J)+'/theta.csv')
	elif len(theta0)==0:
		theta0 = np.random.rand(n+l*n)*2*np.pi
	if gpu:
		return minimize(XY_E_from_theta_lc_gpu,theta0,args=(n,Jz,Jxy,jac),method=method,jac=jac)
	else:
		return minimize(XY_E_from_theta_lc,theta0,args=(n,Jz,Jxy,jac),method=method,jac=jac)


	
def optimize_machine(backend_name,n,l,hx,hz,shots=1024,maxiter = 2000, c0 = 0.6283185307179586, c1 = 0.1, c2 = 0.602, c3 = 0.101, c4 = 0,ansatz='ALAy',symm=True,include_exact=True,gpu=False,readout_mitigate=True):
	from energy_evaluation import Ising_E_from_theta_machine
	from qiskit.aqua.components.optimizers import SPSA
	
	if symm:
		numParams = 2*(l+1)
	elif not symm:
		numParams = n*(l+1)
	
	
	def machine_cost_fn(theta):
		num_thetas = len(theta)//numParams
		theta_reshape = []
		for i in range(num_thetas):
			theta_reshape.append(theta[i*numParams:(i+1)*numParams])
		
		if symm:
			theta_reshape = [[ theta_i[theta_ALAy_to_symm(which_theta,n)] for which_theta in range(n*(l+1))] for theta_i in theta_reshape]
			
		if include_exact and not gpu:
			E_exact = [ising_E_from_theta_lc(theta_i,n,hx,hz,False,False) for theta_i in theta_reshape]
		elif include_exact and gpu:
			E_exact = [ising_E_from_theta_lc_gpu(theta_i,n,hx,hz,False,False) for theta_i in theta_reshape]
		else:
			E_exact = []
		
		E,dE = Ising_E_from_theta_machine(n,theta_reshape,backend_name,['VQE'],shots,hx,hz,E_exact,[],symm,readout_mitigate=readout_mitigate,input_condensed_theta=False)
		
		return E
	
	
	theta0 = np.random.rand(numParams)*2*np.pi
	optimizer = SPSA(maxiter=maxiter, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4)
	optimizer.set_max_evals_grouped(2)
	opt = optimizer.optimize(numParams, machine_cost_fn, initial_point=theta0 )
	return opt



def saveParams(theta,E,directory):
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('made directory')
	if not os.path.isfile(directory+'/E.csv'):
		with open(directory+'/E.csv','w') as fp:
			pass
	if not os.path.isfile(directory+'/theta.csv'):
		with open(directory+'/theta.csv','w') as fp:
			pass
	if not os.path.isfile(directory+'/E_all.csv'):
		with open(directory+'/E_all.csv','w') as fp:
			pass
	
	with open(directory+'/E_all.csv', 'a', newline = '') as csvfile:
		jobwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		jobwriter.writerow([E])
		
	E_previous = np.genfromtxt(directory+'/E.csv')
	if E_previous.size == 0 or E < E_previous:
		np.savetxt(directory+'/E.csv',[E],fmt='%.18f')
		np.savetxt(directory+'/theta.csv',theta,delimiter=',',fmt='%.18f')
		
