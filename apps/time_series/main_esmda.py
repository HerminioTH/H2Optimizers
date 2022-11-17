import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.join("..", "..", "libs"))
from Optimizers import ESMDA

def apply_dark_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#37474fff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.spines['bottom'].set_color('white')
			ax.spines['top'].set_color('white')
			ax.spines['right'].set_color('white')
			ax.spines['left'].set_color('white')
			ax.tick_params(axis='x', colors='white', which='both')
			ax.tick_params(axis='y', colors='white', which='both')
			ax.yaxis.label.set_color('white')
			ax.xaxis.label.set_color('white')
			ax.title.set_color('white')
			ax.set_facecolor("#424242ff")
			ax.set_axisbelow(True)
			ax.grid(True, color='#5a5a5aff')

def plot_results(my_function, inputs_true, optimizer):
	fig, axes = plt.subplots(1, 4, figsize=(18, 5))
	fig.subplots_adjust(top=0.895, bottom=0.125, left=0.045, right=0.985, hspace=0.2, wspace=0.2)

	# Make predictions with M_post 
	DPost = optimizer.compute_predictions()

	# Compute true prediction
	y_true = my_function.eval(inputs_true)

	# Compute mean parameters of M_post
	mean_M_post = []
	for i in range(optimizer.M_post.shape[0]):
		mean_M_post.append(optimizer.M_post[i,:].mean())

	for i in range(DPost.shape[1]):
		axes[0].plot(my_function.t, DPost[:,i], "-", color="0.5", alpha=0.2)
	axes[0].plot(my_function.t, DPost[:,0], "-", color="0.5", alpha=0.2, label="Simulations")
	axes[0].plot(my_function.t, y_true, "-", color="gold", label="True")
	axes[0].plot(my_function.t, my_function.eval(mean_M_post), "-", color="lightcoral", label="Mean")
	axes[0].set_xlabel("Time", size=14, fontname="serif")
	axes[0].set_ylabel("y(t)", size=14, fontname="serif")
	axes[0].set_title(r"$y(t)=A - \frac{B}{C}t^2 + Csin\left(\frac{t^2}{5}\right) + 100\frac{B}{C}cos\left(\frac{t}{2}\right)$", size=14, fontname="serif")
	axes[0].grid(True)
	axes[0].legend(loc=0, shadow=True, fancybox=True)

	n_bins = 20
	for i in range(1, 4):
		axes[i].hist(optimizer.M_prior[i-1,:], bins=n_bins, color="gray", alpha=0.5, ec="black", label="Prior")
		axes[i].hist(optimizer.M_post[i-1,:], bins=n_bins, color="lightblue", ec="black", label="Post")
		axes[i].grid(True)
		axes[i].legend(loc=0, shadow=True, fancybox=True)

	y_min, y_max = axes[1].get_ylim()
	A_true = inputs_true[0]
	axes[1].plot([A_true, A_true], [y_min, y_max/1.5], "--", color="lightcoral")
	axes[1].text(A_true, y_max/1.4, r"$A_{true}$", color="lightcoral", size=16, fontname="serif", rotation=90)

	y_min, y_max = axes[2].get_ylim()
	B_true = inputs_true[1]
	axes[2].plot([B_true, B_true], [y_min, y_max/1.5], "--", color="lightcoral")
	axes[2].text(B_true, y_max/1.4, r"$B_{true}$", color="lightcoral", size=16, fontname="serif", rotation=90)

	y_min, y_max = axes[3].get_ylim()
	C_true = inputs_true[2]
	axes[3].plot([C_true, C_true], [y_min, y_max/1.5], "--", color="lightcoral")
	axes[3].text(C_true, y_max/1.4, r"$C_{true}$", color="lightcoral", size=16, fontname="serif", rotation=90)

	axes[1].set_xlabel("Coefficient A", size=14, fontname="serif")
	axes[2].set_xlabel("Coefficient B", size=14, fontname="serif")
	axes[3].set_xlabel("Coefficient C", size=14, fontname="serif")

	# print(axes[1].get_ylim())

	apply_dark_theme(fig, axes, transparent=False)
	plt.show()


def main():
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
			return A - B/C*self.t**2 + C*np.sin(0.2*self.t**2) + 100.0*B/C*np.cos(0.5*self.t)

	# Create time vector	
	n_steps = 500
	final_time = 15
	time = np.linspace(-final_time, final_time, n_steps)

	# Generate the prior ensemble
	n_samples = 500
	M_prior = generate_M_prior(n_samples)

	# Choose true input values
	A_true = -20.0
	B_true = 200.0
	C_true = 150.0

	# Create the observed data
	my_function = MyFunction(time)
	y_true = my_function.eval([A_true, B_true, C_true])

	# Apply ES-MDA
	optimizer = ESMDA(my_function.eval, M_prior, y_true, eta=0.01, qsi=0.99, alpha=4, max_ite=50, tol=1e-4, verbose=1)
	optimizer.run()

	# Plot results
	plot_results(my_function, [A_true, B_true, C_true], optimizer)

if __name__ == '__main__':
	main()