import numpy as np
from library import *
from datetime import datetime
from energy_evaluation import read_from_tags
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
## Next, to compare the mitigation methods, submit ansatz circuits with the optimized parameters, in addition to \theta=0 circuits.


# use something like the following. It will need to be modified for your directory and whether you imposed permutation symmetry
def submit_saved_params(n,l,hx,backend_name,hz=0.1,rand_compile=True,noise_scale=1):
	from energy_evaluation import submit_ising, submit_ising_symm
	
	my_directory = '/your_directory/saved_parameters/'
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
				submit_ising(n,theta,backend_name,shots=1024,hx=hxi,hz=hz,E=E,rand_compile=rand_compile,noise_scale=noise_scale)
			elif symm:
				submit_ising_symm(n,theta,backend_name,shots=8192,hx=hxi,hz=hz,E=E,input_condensed_theta=False,rand_compile=rand_compile,noise_scale=noise_scale)
				
def submit_zero_calibration(n,l,backend_name,rand_compile=True,noise_scale=1):
	from energy_evaluation import all_ising_Paulis_symm, submit_circuits
	
	if not hasattr(l,'__iter__'):
		l = [l]
	
	whichPauli = all_ising_Paulis_symm(n)
	for li in l:
		theta = np.zeros(n*(li+1))
		submit_circuits(theta,whichPauli,backend_name,tags=['zero_theta_calibration'],shots=8192,rand_compile=rand_compile,noise_scale=noise_scale)

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
	from energy_evaluation import ising_energy_from_job
	E_exact = read_from_tags('E',job.tags())
	E_meas, dE_meas = ising_energy_from_job(job,readout_mitigate,readout_calibration_job)
	print('E_meas = '+str(E_meas))
	print('dE_meas = '+str(dE_meas))
	damping = E_meas/E_exact
	d_damping = abs(dE_meas/E_exact)
	return damping, d_damping

## We use several methods of predicting the damping factor:




def plot_figs(backend,n=20,hx=1.5,hz=0.1,l_all=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50],readout_mitigate=True,plot_ZNE=False,load_from_saved=False,threshold=0.1,plot_ZNE_calib=False,plot_from_pert=False):
	import matplotlib.pyplot as plt
	from matplotlib import container
	from matplotlib import colors
	import pickle
	
	## first, retrieve the data:
	if readout_mitigate:
		filename = backend.name()+'_n'+str(n)+".p"
	else:
		filename = backend.name()+'_n'+str(n)+"_no_readout_mitigation.p"
	
	if load_from_saved:
		damping, d_damping, rel_error, d_rel_error = pickle.load( open( filename, "rb" ) )
	else:
		damping = {}
		d_damping = {}
		rel_error = {}
		d_rel_error = {}
		methods = ['raw','from pert','from small $l$',r'$\theta = 0$']
		for method in methods:
			damping[method] = np.empty(len(l_all))
			damping[method][:] = np.nan
			d_damping[method] = np.empty(len(l_all))
			d_damping[method][:] = np.nan
		methods_ZNE = ['ZNE',r'$\theta = 0$ + ZNE first',r'$\theta = 0$ + ZNE last']
		rel_error = {}
		d_rel_error = {}
		for method in methods + methods_ZNE:
			rel_error[method] = np.empty(len(l_all))
			rel_error[method][:] = np.nan
			d_rel_error[method] = np.empty(len(l_all))
			d_rel_error[method][:] = np.nan
		
		
		fit_shallow = small_l_fit(backend,n,hx,hz,max_l=15,readout_mitigate=readout_mitigate)
		
		for _ in range(len(l_all)):
			l = l_all[_]
			print('starting l = '+str(l))
			if l > 0:
				limit = 3
			else:
				limit = 1
			jobs = backend.jobs(limit=limit,job_tags=['n = '+str(n),'l = '+str(l),'hx = '+str(hx),'hz = '+str(hz)],job_tags_operator='AND')
			jobs_calib = backend.jobs(limit=limit,job_tags=['zero_theta_calibration','n = '+str(n),'l = '+str(l)],job_tags_operator='AND')
			for job in jobs:
				if read_from_tags('noise_scale',job.tags()) == 1.0:
					break
			for job_calib in jobs_calib:
				if read_from_tags('noise_scale',job_calib.tags()) == 1.0:
					break
			damping['raw'][_], d_damping['raw'][_] = damping_from_job(job,readout_mitigate)
			damping['from pert'][_], d_damping['from pert'][_] = damping_est_pert(job,readout_mitigate,plot=plot_from_pert,damping1_5=damping['raw'][_],d_damping1_5=d_damping['raw'][_])
			damping['from small $l$'][_], d_damping['from small $l$'] = pred_from_fit(l,fit_shallow)
			damping[r'$\theta = 0$'][_], d_damping[r'$\theta = 0$'][_] = damping_from_zero_theta_energy(job_calib,hx,hz,readout_mitigate)
			if l > 0:
				rel_error['ZNE'][_], d_rel_error['ZNE'][_] = ZNE(jobs,readout_mitigate=readout_mitigate,plot=plot_ZNE)
				rel_error[r'$\theta = 0$ + ZNE last'][_], d_rel_error[r'$\theta = 0$ + ZNE last'][_] = damping_zero_theta_ZNE(jobs,jobs_calib,order='extrapolate_last',readout_mitigate=readout_mitigate,plot=plot_ZNE_calib)
				rel_error[r'$\theta = 0$ + ZNE first'][_], d_rel_error[r'$\theta = 0$ + ZNE first'][_] = damping_zero_theta_ZNE(jobs,jobs_calib,order='extrapolate_first',readout_mitigate=readout_mitigate,plot=plot_ZNE_calib)
				for method in rel_error:
					rel_error[method][_] -= 1
		for method in damping:
			if (method != 'raw' and method != 'ZNE'):
				rel_error[method] = damping['raw']/damping[method] - 1
				d_rel_error[method] = np.sqrt( (d_damping['raw']/damping[method])**2 + (damping['raw']*d_damping[method]/damping[method]**2))
			elif method == 'raw':
				rel_error[method] = damping[method] - 1
				d_rel_error[method] = d_damping[method]
		
		pickle.dump( (damping, d_damping, rel_error, d_rel_error), open( filename, "wb" ) )
	
	
	## now plot
	markers = ['o','v','^','<','>','s','P','*','+','x','D']
	marker_i = 0
	
	### damping:
	
	fig, ax = plt.subplots()
	for method in damping:
		if method == 'raw':
			plt.errorbar(l_all,damping[method],d_damping[method],label='true damping factor',linewidth=3,capsize=4,fmt=markers[marker_i])
		else:
			plt.errorbar(l_all,damping[method],d_damping[method],label='predicted, '+method,linewidth=3,capsize=4,fmt=markers[marker_i]+'-')
		marker_i += 1
	plt.xlabel('number of ansatz layers',fontsize = 20)
	plt.ylabel('actual or predicted damping factor',fontsize = 18)
	# removing error bars from legend using https://swdg.io/2015/errorbar-legends/
	handles, labels = ax.get_legend_handles_labels()

	new_handles = []

	for h in handles:
		#only need to edit the errorbar legend entries
		if isinstance(h, container.ErrorbarContainer):
			new_handles.append(h[0])
		else:
			new_handles.append(h)

	ax.legend(new_handles, labels,loc='best',prop={'size': 11})
	plt.ylim((1e-2,1))
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.tick_params(axis='both', which='minor', labelsize=15)
	plt.title(backend.name(),fontsize=20)
	plt.yscale('log')
	
	fig.tight_layout()
	
	
	### relative error:
	marker_i = 0
	fig, ax = plt.subplots()
	for method in rel_error:
		plt.errorbar(l_all,rel_error[method],d_rel_error[method],label=method,linewidth=3,capsize=4,fmt=markers[marker_i]+'-')
		marker_i += 1
	plt.xlabel('number of ansatz layers',fontsize = 20)
	plt.ylabel('relative error',fontsize = 18)
	
	# removing error bars from legend using https://swdg.io/2015/errorbar-legends/
	handles, labels = ax.get_legend_handles_labels()

	new_handles = []

	for h in handles:
		#only need to edit the errorbar legend entries
		if isinstance(h, container.ErrorbarContainer):
			new_handles.append(h[0])
		else:
			new_handles.append(h)

	ax.legend(new_handles, labels,loc='best',prop={'size': 11})
		
	#plt.legend(loc='best',prop={'size': 11})
	plt.plot([min(l_all),max(l_all)],[threshold,threshold],'k--',linewidth=2)
	plt.plot([min(l_all),max(l_all)],[-threshold,-threshold],'k--',linewidth=2)
	plt.ylim((-1,3))
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.tick_params(axis='both', which='minor', labelsize=15)
	plt.title(backend.name(),fontsize=20)
	
	fig.tight_layout()
	
	cmap = colors.ListedColormap(np.array([[255,255,204],[161,218,180],[65,182,196],[34,94,168]])/255)
	scores = {}
	for method in rel_error:
		scores[method] = rel_error_score(rel_error[method],d_rel_error[method],threshold)
	
	fig, ax = plt.subplots()
	im = ax.imshow(list(scores.values()),cmap=cmap)
	# Loop over data dimensions and create text annotations.
	for i in range(len(scores)):
		for j in range(len(l_all)):
			if not np.isnan(list(scores.values())[i][j]):
				text = ax.text(j, i, list(scores.values())[i][j], ha="center", va="center", color="k")
	# We want to show all ticks...
	ax.set_xticks(np.arange(len(l_all)))
	ax.set_yticks(np.arange(len(scores)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(l_all)
	ax.set_yticklabels(list(scores.keys()))

	plt.xlabel('number of ansatz layers',fontsize = 18)
	#plt.ylabel('mitigation method',fontsize = 15)
	
	plt.title(str(n)+' qubits, '+backend.name(),fontsize=15)
	
	
	
	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=11)
	ax.tick_params(axis='x', which='minor', labelsize=11)
	
	fig.tight_layout()
	
	plt.show()
		


# From the perturbative regime:
def damping_est_pert(job,readout_mitigate=True,calibration_job=[],noise_scale=1,plot=False,damping1_5=0,d_damping1_5=0):
	backend = job.backend()
	tags = job.tags()
	n = read_from_tags('n',tags)
	hz = read_from_tags('hz',tags)
	symm = 'symm' in tags
	l = read_from_tags('l',tags)
	
	hx_pert = [0.1,0.2,0.3,0.4,0.5]
	damping_all = []
	d_damping_all = []
	for hx in hx_pert:
		desired_tags = ['Ising','l = '+str(l),'hx = '+str(hx),'n = '+str(n),'hz = '+str(hz),'noise_scale = '+str(noise_scale)]
		if symm:
			desired_tags.append('symm')
		job_pert = backend.jobs(limit=1,job_tags=desired_tags,job_tags_operator='AND')[0]
		damping_i, d_damping_i = damping_from_job(job_pert,readout_mitigate,calibration_job)
		damping_all.append(damping_i)
		d_damping_all.append(d_damping_i)
	
	damping = np.mean(damping_all)
	d_damping = np.std(damping_all)/np.sqrt(len(hx_pert))
	
	if plot:
		import matplotlib.pyplot as plt
		hx = hx_pert + [1.5]
		damping_all.append(damping1_5)
		d_damping_all.append(d_damping1_5)
		plt.errorbar(hx,damping_all,d_damping_all,fmt='.',capsize=4,label='observed damping factors')
		plt.plot([min(hx),max(hx)],[damping,damping],'k')
		plt.plot([min(hx),max(hx)],[damping+d_damping,damping+d_damping],'k--')
		plt.plot([min(hx),max(hx)],[damping-d_damping,damping-d_damping],'k--')
		plt.legend(loc='best',prop={'size':15})
		plt.xlabel('$h_x$', fontsize=20)
		plt.ylabel('damping factor',fontsize=20)
		ax = plt.gca()
		ax.tick_params(axis='both', which='major', labelsize=15)
		ax.tick_params(axis='both', which='minor', labelsize=15)
		if l != 1:
			plt.title(backend.name()+', '+str(l)+' ansatz layers',fontsize=20)
		else:
			plt.title(backend.name()+', '+str(l)+' ansatz layer',fontsize=20)
		plt.tight_layout()
		plt.show()
	return damping, d_damping
	


# ZNE:
	
def ZNE(jobs,readout_mitigate=True,plot=True):
	import matplotlib.pyplot as plt
	from matplotlib import container
	scales = [read_from_tags('noise_scale',j.tags()) for j in jobs]
	dampings = []
	d_dampings = []
	for job in jobs:
		damping, d_damping = damping_from_job(job,readout_mitigate=readout_mitigate)
		dampings.append(damping)
		d_dampings.append(d_damping)
	
	from scipy.optimize import curve_fit
	try:
		fit = curve_fit(exp_fit,scales,dampings,p0=[1,0.5],sigma=d_dampings,absolute_sigma=True)
		failed = False
	except:
		print('error: fit failed')
		failed = True
	if plot:
		fig, ax = plt.subplots()
		print('scales = '+str(scales))
		print('dampings = '+str(dampings))
		plt.errorbar(scales,dampings,d_dampings,label='measured energy/exact energy',linewidth=3,capsize=4)
		if not failed:
			plt.plot(np.linspace(0,max(scales),100),exp_fit(np.linspace(0,max(scales),100),fit[0][0], fit[0][1]),label='exponential fit')
		plt.xlabel('noise scale',fontsize = 18)
		plt.ylabel('energy/exact energy',fontsize = 18)
		plt.xlim([0,max(scales)])
		dampings = np.array(dampings)
		d_dampings = np.array(d_dampings)
		plt.ylim([min(dampings-d_dampings),max(max(dampings+d_dampings), exp_fit(0,fit[0][0], fit[0][1]), 1)])
		plt.yscale('log')
		ax.tick_params(axis='both', which='major', labelsize=15)
		ax.tick_params(axis='both', which='minor', labelsize=15)
		
		# removing error bars from legend using https://swdg.io/2015/errorbar-legends/
		handles, labels = ax.get_legend_handles_labels()

		new_handles = []

		for h in handles:
			#only need to edit the errorbar legend entries
			if isinstance(h, container.ErrorbarContainer):
				new_handles.append(h[0])
			else:
				new_handles.append(h)

		ax.legend(new_handles, labels,loc='best',prop={'size': 11})
		fig.tight_layout()
		plt.show()
	if failed:
		return float('nan'), float('nan')
	else:
		return pred_from_fit(0,fit)
	
	
def ZNE_zero_theta(jobs,hx,hz,readout_mitigate=True,plot=True):
	import matplotlib.pyplot as plt
	scales = [read_from_tags('noise_scale',j.tags()) for j in jobs]
	dampings = []
	d_dampings = []
	for job in jobs:
		damping, d_damping = damping_from_zero_theta_energy(job,hx,hz,readout_mitigate=readout_mitigate)
		dampings.append(damping)
		d_dampings.append(d_damping)
	
	from scipy.optimize import curve_fit
	try:
		fit = curve_fit(exp_fit,scales,dampings,p0=[1,0.5],sigma=d_dampings,absolute_sigma=True)
	except:
		print('error: fit failed')
		return float('nan'), float('nan')
	if plot:	
		plt.errorbar(scales,dampings,d_dampings,label='data')
		plt.plot(np.linspace(0,max(scales),100),exp_fit(np.linspace(0,max(scales),100),fit[0][0], fit[0][1]),label='fit')
		plt.legend(loc='best')
		plt.xlabel('noise scale')
		plt.ylabel('damping factor')
		plt.xlim([0,max(scales)])
		plt.show()
	
	return pred_from_fit(0,fit)
	

def damping_zero_theta_ZNE(jobs_ZNE,jobs_ZNE_zero_theta,order='extrapolate_last',readout_mitigate=True,plot=True):
	hx = read_from_tags('hx',jobs_ZNE[0].tags())
	hz = read_from_tags('hz',jobs_ZNE[0].tags())
	
	if order == 'extrapolate_first':
		damping, d_damping = ZNE(jobs_ZNE,readout_mitigate=readout_mitigate,plot=plot)
		damping_zero_theta, d_damping_zero_theta = ZNE_zero_theta(jobs_ZNE_zero_theta,hx,hz,readout_mitigate=readout_mitigate,plot=plot)
		return damping/damping_zero_theta, np.sqrt( (d_damping/damping_zero_theta)**2 + (damping*d_damping_zero_theta/damping_zero_theta**2)**2)
	elif order == 'extrapolate_last':
		dampings = []
		d_dampings = []
		scales = []
		for i in range(len(jobs_ZNE)):
			job = jobs_ZNE[i]
			job_calib = jobs_ZNE_zero_theta[i]
			scales.append(read_from_tags('noise_scale',job.tags()))
			damping, d_damping = damping_from_job(job,readout_mitigate=readout_mitigate)
			damping_calib, d_damping_calib = damping_from_zero_theta_energy(job_calib,hx,hz,readout_mitigate=readout_mitigate)
			dampings.append(damping/damping_calib)
			d_dampings.append(np.sqrt( (d_damping/damping_calib)**2 + (damping*d_damping_calib/damping_calib**2)**2))
			print('scale = '+str(scales[-1]))
			print('damping_i = '+str(dampings[-1]))
			print('d_damping_i = '+str(d_dampings[-1]))
		
		from scipy.optimize import curve_fit
		try:
			fit = curve_fit(exp_fit,scales,dampings,p0=[1,0.5],sigma=d_dampings,absolute_sigma=True)
		except:
			print('error: fit failed')
			return float('nan'), float('nan')
		if plot:	
			import matplotlib.pyplot as plt
			plt.errorbar(scales,dampings,d_dampings,label='data')
			plt.plot(np.linspace(0,max(scales),100),exp_fit(np.linspace(0,max(scales),100),fit[0][0], fit[0][1]),label='fit')
			plt.legend(loc='best')
			plt.xlabel('noise scale')
			plt.ylabel('damping factor')
			plt.xlim([0,max(scales)])
			plt.show()
		
		return pred_from_fit(0,fit)
	


# from small l:

def exp_fit(l,A,b):
	return A*np.exp(-b*l)

def small_l_fit(backend,n,hx,hz,max_l=15,readout_mitigate=True,noise_scale=1):
	from scipy.optimize import curve_fit
	l_all = range(0,max_l+1)
	damping_all = []
	d_damping_all = []
	for l in l_all:
		desired_tags = ['Ising','l = '+str(l),'hx = '+str(hx),'n = '+str(n),'hz = '+str(hz)]
		jobs = backend.jobs(limit=3,job_tags=desired_tags,job_tags_operator='AND')
		for job in jobs:
			if read_from_tags('noise_scale',job.tags()) == noise_scale:
				break
		damping_i, d_damping_i = damping_from_job(job,readout_mitigate)
		damping_all.append(damping_i)
		d_damping_all.append(d_damping_i)
		
	fit_shallow = curve_fit(exp_fit,l_all,damping_all,p0=[1,0.5],sigma=d_damping_all,absolute_sigma=True)
	
	return fit_shallow
	
def pred_from_fit(l,fit,size=100000):
	rng = np.random.default_rng()
	params = rng.multivariate_normal(fit[0],fit[1],size=size)
	est = exp_fit(l,params[:,0],params[:,1])
	return np.mean(est), np.std(est)






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


	
def damping_from_zero_theta_energy(zero_calib_job,hx,hz,readout_mitigate=True,readout_calibrate_job=[]):
	from energy_evaluation import ising_energy_from_job, energy_from_job
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
				qc_i = ansatz_circuit(theta_i,pauli,rand_compile=False,noise_scale=1)
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
	
	
################# The following is not finished: ##########################



## more careful readout mitigation:

def qubits_measured_from_job(job):
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
	
	save_dir = '/your_directory/results/damping_factors/'+backend_name+'/n'+str(n)+'/'
	
	
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


################## randomized measurement of Tr[\rho^2]

def matrix_to_euler_angles(u):
	## u is an arbitrary 2x2 unitary matrix. Returns the Euler angles that can be fed into a UGate (https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html) to reproduce the input matrix up to a global phase.
	
	# first divide out phase of 00 element
	if abs(u[0,0]) > 1e-5: # if 00 element is nonzero
		phase = u[0,0]/abs(u[0,0])
		u = u/phase
	#now extract the parameters
	theta = 2*np.arccos(u[0,0].real)
	if abs(theta) > 1e-5: # if off-diagonal elements are nonzero
		phi = np.angle(u[1,0])
		lam = np.angle(-u[0,1])
	else:
		phi = np.angle(u[1,1])
		lam = 0
		
	return theta, phi, lam
	
def Harr_random_Euler_angles():
	from scipy.stats import unitary_group
	u = unitary_group.rvs(2)
	return matrix_to_euler_angles(u)
	
	
	
def ansatz_circuit_with_random_u(theta,whichPauli):
	from energy_evaluation import ansatz_circuit
	from scipy.stats import unitary_group
	
	h = np.array([[1,1],[1,-1]])/np.sqrt(2)
	def ry(th):
		return np.array( [[np.cos(th/2), -np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]])
	sdg = np.array([[1,0],[0,-1j]])
	
	qc = ansatz_circuit(theta,whichPauli,False,False,rand_compile=False)
	n = len(whichPauli)
	l = len(theta)//n - 1
	qubitsMeasured = [i for i in range(n) if whichPauli[i] > 0]
	num_qubits_measured = len(qubitsMeasured)
	for i in range(num_qubits_measured):
		q = qubitsMeasured[i]
		if whichPauli[q] == 1:
			u = unitary_group.rvs(2)@h@ry(theta[n*l + q])
		elif whichPauli[q] == 2:
			u = unitary_group.rvs(2)@h@sdg@ry(theta[n*l + q])
		elif whichPauli[q] == 3:
			u = unitary_group.rvs(2)@ry(theta[n*l + q])
		
		th, phi, lam = matrix_to_euler_angles(u)
		qc.u(th,phi,lam,q)
		qc.measure(q,i)
	return qc
	


	
def submit_circuits_with_random_u(theta,whichPauli_all,backend_name,num_random_u=10,tags=[],shots=8192,configs_all_terms=[],include_original_circuit=True):
	from qiskit import IBMQ, execute
	from energy_evaluation import ansatz_circuit, cycle_QuantumCircuit, load_qubit_map, pick_config
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
			qc = ansatz_circuit(th_i,whichPauli,True,rand_compile=False)
			for config in configs_term:
				if include_original_circuit:
					qc_all.append( cycle_QuantumCircuit(qc,config))
				for _ in range(num_random_u):
					qc_u = ansatz_circuit_with_random_u(th_i,whichPauli)
					qc_all.append( cycle_QuantumCircuit(qc_u,config))

	
	
	tags += ['n = '+str(n),'l = '+str(l),'theta = '+str(theta),'configs = '+str(configs_all_terms),'whichPauli = '+str(whichPauli_all),'rand_u','num_random_u = '+str(num_random_u),'include_original_circuit = '+str(include_original_circuit)]
	
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



def submit_ising_symm_with_random_u(n,theta,backend_name,tags=[],shots=8192,hx=1.5,hz=0.1,E=[],configs_all_terms=[],input_condensed_theta=True,num_random_u=10,include_original_circuit=True):

	from energy_evaluation import all_ising_Paulis_symm
	
	multi_theta = len(np.shape(theta)) > 1
	if not multi_theta:
		theta = [theta]
	
	if input_condensed_theta:
		l = len(theta[0])//2 - 1
		theta_full = [[ theta_i[theta_ALAy_to_symm(which_theta,n)] for which_theta in range(n*(l+1))] for theta_i in theta]
		theta = theta_full
	
	return submit_circuits_with_random_u(theta,all_ising_Paulis_symm(n),backend_name,num_random_u,tags+['Ising','symm','hx = '+str(hx),'hz = '+str(hz),'E = '+str(E)],shots,configs_all_terms,include_original_circuit)



def hamming(s1,s2):
	D = 0
	for i in range(len(s1)):
		if s1[i] != s2[i]:
			D += 1
	return D


def tr_rho2(counts_all):
	# computes Eq. 2 from doi:10.1126/science.aau4963
	
	num_random_u = len(counts_all)
	n_meas = len( list(counts_all[0].keys())[0] )
	
	shots = 0
	for s in counts_all[0]:
		shots += counts_all[0][s]
	
	tr = 0
	d_tr2 = 0
	
	
	for counts in counts_all:
		for s1 in counts:
			for s2 in counts:
				P1 = counts[s1]/shots
				P2 = counts[s2]/shots
				D = hamming(s1,s2)
				tr += 2**n_meas * (-2)**(-D) * P1 * P2
				dP1 = np.sqrt(P1*(1-P1)/shots)
				dP2 = np.sqrt(P2*(1-P2)/shots)
				if s1 != s2:
					d_tr2 += (2**n_meas * (-2)**(-D) * dP1 * P2)**2 + (2**n_meas * (-2)**(-D) * P1 * dP2)**2
				elif s1 == s2:
					d_tr2 += (2 * 2**n_meas * (-2)**(-D) * dP1 * P1)**2 
				
	tr = tr/num_random_u
	d_dr = np.sqrt(d_tr2)/num_random_u
				
	return tr, d_tr


