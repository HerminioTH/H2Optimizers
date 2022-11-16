import numpy as np
from functools import partial

class PSO():
	def __init__(self, func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
			        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
			        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
			        particle_output=False, history=False, output_folder=False):
		pass

	def run(self):
		pass


class ESMDA():
	def __init__(self, objective_function, M_prior, y_true, eta=0.001, qsi=0.99, alpha=4):
		self.M_prior = M_prior
		self.y_abs = np.abs(y_true)
		self.eta = eta
		self.qsi = qsi
		self.ce_diag = self.eta*self.y_abs
		self.c_ev = np.sqrt(self.ce_diag.reshape([y_true.shape[0], 1]))
		self.alphas = [alpha for i in range(alpha)]
		self.obj = objective_function
		self.M_post = self.M_prior.copy()

	def run(self):
		s_diag = np.sqrt(self.ce_diag)
		s_inv_diag = np.power(s_diag, -1)
		INd = np.eye(len(self.y_abs))

		for alpha in self.alphas:
			print(alpha)
			D = self.__compute_predictions()
			# print(self.M_post)
			print(D.std())
			self.__update_M_post(np.abs(D), alpha)

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
		Cinv = Cinv/(self.c_ev@self.c_ev.T)
		Dobs = self.y_abs.reshape([Nd,1]) + self.c_ev*np.random.randn(Nd,Ne) #Nd x Ne
		self.M_post += Cmd@Cinv@(Dobs - D) #(Nm x Nd)x(Nd x Nd)x(Nd x Ne) = Nm x Ne

	def __compute_predictions(self):
		D = np.zeros((len(self.y_abs), self.M_post.shape[1]))
		for i in range(self.M_post.shape[1]):
			inputs = self.M_post[:,i]
			# print(i, inputs)
			D[:,i] = self.obj(inputs)
		return D



if __name__ == '__main__':
	def generate_M_prior(n_samples):
		np.random.seed(42)
		A = np.random.normal(loc=150, scale=50, size=n_samples).reshape((1, n_samples))
		B = np.random.normal(loc=150, scale=50, size=n_samples).reshape((1, n_samples))
		C = np.random.normal(loc=150, scale=50, size=n_samples).reshape((1, n_samples))
		M_prior = np.concatenate((A, B, C), axis=0)
		return M_prior

	class MyFunction():
		def __init__(self, t):
			self.t = t

		def eval(self, inputs):
			A, B, C = inputs[0], inputs[1], inputs[2]
			return A + B*self.t + C*np.sin(self.t**2)

	# Create time vector	
	n_steps = 10
	final_time = 5
	time = np.linspace(0, final_time, n_steps)

	# Generate the prior ensemble
	n_samples = 25
	M_prior = generate_M_prior(n_samples)

	# Choose true input values
	A_true = 200.0
	B_true = 200.0
	C_true = 150.0

	# Create the observed data
	my_function = MyFunction(time)
	y_true = my_function.eval([A_true, B_true, C_true])

	# Apply ES-MDA
	opt = ESMDA(my_function.eval, M_prior, y_true, eta=0.001, qsi=0.99, alpha=4)
	opt.run()


	print()
	print("A_true = %.2f"%A_true)
	print("B_true = %.2f"%B_true)
	print("C_true = %.2f"%C_true)
	print()

	A_mean = opt.M_post[0,:].mean()
	B_mean = opt.M_post[1,:].mean()
	C_mean = opt.M_post[2,:].mean()
	print("A = %.2f"%A_mean)
	print("B = %.2f"%B_mean)
	print("C = %.2f"%C_mean)



