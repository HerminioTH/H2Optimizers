import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import sys
import os
sys.path.append(os.path.join("..", "..", "..", "MechanicsAnalytic", "libs"))
sys.path.append(os.path.join("..", "..", "libs"))
from Optimizers import ESMDA
from AnalyticSolutions import *

def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

class MyModel():
		def __init__(self, settings, scaler):
			self.settings = settings
			self.scaler = scaler

		# def eval(self, M):
		# 	# M = self.scaler.inverse_transform(M.T).T
		# 	y = []
		# 	for m in M.T:
		# 		# Update settings
		# 		self.extract_props_to_settings(m)

		# 		# Initialize viscoelastic model
		# 		model_cr = Creep(self.settings)
		# 		model_ve = ViscoElastic(self.settings)

		# 		# Compute strains
		# 		model_cr.compute_strains()
		# 		model_ve.compute_strains()

		# 		# Total strain
		# 		eps_tot = model_ve.eps + model_cr.eps
		# 		y.append(np.concatenate((eps_tot[1:,0,0], eps_tot[1:,2,2]), axis=0))
		# 	return np.array(y)

		def eval(self, m):
			# Update settings
			self.extract_props_to_settings(m)

			# Initialize viscoelastic model
			model_cr = Creep(self.settings)
			model_ve = ViscoElastic(self.settings)

			# Compute strains
			model_cr.compute_strains()
			model_ve.compute_strains()

			# Total strain
			eps_tot = model_ve.eps + model_cr.eps
			y = np.concatenate((eps_tot[1:,0,0], eps_tot[1:,2,2]), axis=0)
			return y

		def extract_props_to_settings(self, m):
			self.settings["elasticity"]["E"] = m[0]
			self.settings["elasticity"]["nu"] = m[1]
			self.settings["viscoelasticity"]["E"][0] = m[2]
			self.settings["viscoelasticity"]["eta"][0] = m[3]
			self.settings["viscoelasticity"]["E"][1] = m[4]
			self.settings["viscoelasticity"]["eta"][1] = m[5]

def main():
	# Read settings
	settings = read_json("settings.json")

	# Build input parameters
	E = settings["elasticity"]["E"]
	nu = settings["elasticity"]["nu"]
	E0 = settings["viscoelasticity"]["E"][0]
	eta0 = settings["viscoelasticity"]["eta"][0]
	E1 = settings["viscoelasticity"]["E"][1]
	eta1 = settings["viscoelasticity"]["eta"][1]
	M_true = np.array([[E, nu, E0, eta0, E1, eta1]]).T

	# Generate the prior ensemble
	n_samples = 200
	np.random.seed(42)
	E_prior = np.clip(np.random.normal(loc=10*GPa, scale=7*GPa, size=n_samples).reshape((1, n_samples)), 0.5*GPa, 50*GPa)
	nu_prior = np.clip(np.random.normal(loc=0.25, scale=0.15, size=n_samples).reshape((1, n_samples)), 0.0, 0.499)
	E0_prior = np.clip(np.random.normal(loc=10*GPa, scale=7*GPa, size=n_samples).reshape((1, n_samples)), 0.5*GPa, 50*GPa)
	eta0_prior = np.clip(np.random.normal(loc=2e15, scale=9e14, size=n_samples).reshape((1, n_samples)), 9e13, 9e15)
	E1_prior = np.clip(np.random.normal(loc=10*GPa, scale=7*GPa, size=n_samples).reshape((1, n_samples)), 0.5*GPa, 50*GPa)
	eta1_prior = np.clip(np.random.normal(loc=2e15, scale=9e14, size=n_samples).reshape((1, n_samples)), 9e13, 9e15)
	M_prior = np.concatenate((E_prior, nu_prior, E0_prior, eta0_prior, E1_prior, eta1_prior), axis=0)

	# Define scaler
	scaler = MinMaxScaler(feature_range=(100, 200))

	# Fit scaler
	scaler.fit(M_prior.T)

	# Transform M_prior and M_true
	M_prior_scl = scaler.transform(M_prior.T).T
	M_true_scl = scaler.transform(M_true.T).T
	# M_prior_scl = M_prior
	# M_true_scl = M_true

	# Compute y_true
	model = MyModel(settings, scaler)
	y_true = model.eval(M_true_scl.flatten())

	# Apply ES-MDA
	optimizer = ESMDA(model.eval, M_prior_scl, y_true, eta=0.001, qsi=0.99, alpha=20, verbose=1)
	optimizer.run()

	# Compute mean parameters of M_post
	print(M_true)
	mean_M_post = []
	M_post = scaler.inverse_transform(optimizer.M_post.T).T
	for i in range(M_post.shape[0]):
		mean_M_post.append(M_post[i,:].mean())
		print("%.3e"%M_post[i,:].mean())

	# print(mean_M_post)


if __name__ == '__main__':
	main()