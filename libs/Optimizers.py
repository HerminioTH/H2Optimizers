import numpy as np
from functools import partial

# class PSO():
# 	def __init__(self, func, M_prior, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
# 			        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
# 			        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
# 			        particle_output=False, history=False, output_folder=False):

# 	def run(self):
# 		pass


class ESMDA():
	def __init__(self, func, M_prior, y_true, eta=0.001, qsi=0.99, alpha=4, verbose=0):
		self.obj = func
		self.M_prior = M_prior
		self.y_abs = np.abs(y_true)
		self.eta = eta
		self.qsi = qsi
		self.alpha = alpha
		self.verbose = verbose

		self.__initialize()

	def run(self):
		s_diag = np.sqrt(self.ce_diag)
		s_inv_diag = np.power(s_diag, -1)
		INd = np.eye(len(self.y_abs))

		ite = 0
		while ite <= self.alpha:
			D = self.compute_predictions()
			self.__update_M_post(np.abs(D), self.alpha)
			if self.verbose == 1:
				print(f"{ite}/{self.alpha} - {D.std()}")
			D_old = D.copy()
			ite += 1

	# def find_best(self, D):

	def compute_predictions(self):
		D = np.zeros((len(self.y_abs), self.M_post.shape[1]))
		for i in range(self.M_post.shape[1]):
			inputs = self.M_post[:,i]
			D[:,i] = self.obj(inputs)
		return D

	def __update_M_post(self, D, alpha):
		Nd = self.y_abs.shape[0]
		Ne = D.shape[1]
		dD = np.transpose(D.T - D.mean(axis=1)) #Nd x Ne
		dM = np.transpose(self.M_post.T - self.M_post.mean(axis=1)) #Nm x Ne
		Cmd = dM @ dD.T #Nm x Nd
		dD = dD/self.c_ev
		b = (Ne - 1)*alpha
		if Nd > Ne: #woodbury matrix identity
		    Cinvt = np.linalg.pinv(dD.T@dD + np.eye(Ne)*b)
		    Cinv = (np.eye(Nd) - dD@Cinvt@dD.T)/b
		else:
		    Cinv = np.linalg.pinv(dD@dD.T + np.eye(Nd)*b)
		Cinv = Cinv/self.c_ev_2
		Dobs = self.y_abs.reshape([Nd,1]) + self.c_ev*np.random.randn(Nd,Ne) #Nd x Ne
		self.M_post += Cmd@Cinv@(Dobs - D) #(Nm x Nd)x(Nd x Nd)x(Nd x Ne) = Nm x Ne

	def __initialize(self):
		Nd = self.y_abs.shape[0]
		Ne = self.M_prior.shape[1]
		self.ce_diag = self.eta*self.y_abs
		# print(self.y_abs.shape)
		self.c_ev = np.sqrt(self.ce_diag.reshape([self.y_abs.shape[0], 1]))
		self.c_ev_2 = self.c_ev@self.c_ev.T
		self.D_obs = self.y_abs.reshape([Nd,1]) + self.c_ev*np.random.randn(Nd,Ne) #Nd x Ne
		self.M_post = self.M_prior.copy()


