import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
	# Make predictions with M_post 
	DPost = optimizer.compute_predictions()

	# Compute true prediction
	y_true = my_function.eval(inputs_true)

	# Compute mean parameters of M_post
	mean_M_post = []
	for i in range(optimizer.M_post.shape[0]):
		mean_M_post.append(optimizer.M_post[i,:].mean())

	print(mean_M_post)

	print(optimizer.M_post.shape)

	fig = plt.figure(figsize=(5,4))
	fig.subplots_adjust(top=0.985, bottom=0.075, left=0.085, right=0.970, hspace=0.290, wspace=0.29)
	gs = GridSpec(7, 12)

	ax_hist_h = fig.add_subplot(gs[0:2,1:9])
	ax_scatter = fig.add_subplot(gs[2:7, 1:9])
	ax_hist_v = fig.add_subplot(gs[2:7, 9:12])

	x0_min = min(optimizer.M_prior[0,:].min(), optimizer.M_post[0,:].min())
	x0_max = max(optimizer.M_prior[0,:].max(), optimizer.M_post[0,:].max())
	x1_min = min(optimizer.M_prior[1,:].min(), optimizer.M_post[1,:].min())
	x1_max = max(optimizer.M_prior[1,:].max(), optimizer.M_post[1,:].max())
	n_points = 50
	x0 = np.linspace(x0_min, x0_max, n_points)
	x1 = np.linspace(x1_min, x1_max, n_points)
	X0, X1 = np.meshgrid(x0, x1)
	Z = my_function.eval([X0, X1])
	print(X0.shape)
	print(Z[0].shape)
	ax_scatter.contourf(X0, X1, Z[0], 20, cmap=plt.cm.bone, linestyles=1.0)

	ax_scatter.plot([inputs_true[0]], [inputs_true[1]], "*", color="limegreen", markersize=10, markeredgecolor="white")
	ax_scatter.scatter(optimizer.M_prior[0,:], optimizer.M_prior[1,:], s=15, edgecolors="k", linewidths=1.5, c="gray")
	ax_scatter.scatter(optimizer.M_post[0,:], optimizer.M_post[1,:], s=15, edgecolors="k", linewidths=1.5, c="lightblue")

	ax_hist_h.hist(optimizer.M_prior[0,:], bins=20, color="gray", alpha=1.0, ec="black", label="Prior")
	ax_hist_h.hist(optimizer.M_post[0,:], bins=20, color="lightblue", alpha=1.0, ec="black", label="Post")
	ax_hist_h.tick_params(left=True, right=False, labelleft=True, labelbottom=False, bottom=False)
	ax_hist_h.legend(loc=0, shadow=True, fancybox=True, prop={'size': 6})

	ax_hist_v.hist(optimizer.M_prior[1,:], bins=20, color="gray", alpha=1.0, ec="black", orientation="horizontal", label="Prior")
	ax_hist_v.hist(optimizer.M_post[1,:], bins=20, color="lightblue", alpha=1.0, ec="black", orientation="horizontal", label="Post")
	ax_hist_v.tick_params(left=False, right=False, labelleft=False, labelbottom=True, bottom=True)
	ax_hist_v.legend(loc=0, shadow=True, fancybox=True, prop={'size': 6})

	apply_dark_theme(fig, [ax_scatter, ax_hist_h, ax_hist_v], transparent=False)
	plt.show()


def main():
	# Define function to be minimized
	class Banana():
		def eval(self, x):
			x1 = x[0]
			x2 = x[1]
			f = [x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5]
			# f = [x1**2 + x2**2]
			return np.array(f) # It has to return an array, even if it's a scalar function

	# Generate the prior ensemble
	n_samples = 1000
	np.random.seed(42)
	x1 = np.random.normal(loc=0, scale=0.5, size=n_samples).reshape((1, n_samples))
	x2 = np.random.normal(loc=0, scale=0.5, size=n_samples).reshape((1, n_samples))
	M_prior = np.concatenate((x1, x2), axis=0)

	# Create the observed data
	x_true = [1.0, 1.0]
	banana = Banana()
	y_true = banana.eval(x=x_true)
	print(y_true)

	# Apply ES-MDA
	alpha = 35
	optimizer = ESMDA(banana.eval, M_prior, y_true, eta=0.00000000000001, qsi=0.99, alpha=alpha, verbose=0)
	optimizer.run()

	# Plot results
	plot_results(banana, x_true, optimizer)

if __name__ == '__main__':
	main()