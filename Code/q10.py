import numpy as np
from Code.garcon import Garcon
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gc = Garcon()

class PolyFit(object):
	def __init__(self, X, y, deg):
		self.deg = deg
		self.poly = None
		self.train(X, y)

	def train(self, X, y):
		coeff = np.polyfit(X, y, self.deg)
		self.poly = np.poly1d(coeff)

	def predict(self, X):
		return self.poly(X)

def get_true_y(X, mu=0, sig=1):
	'''
	:param X:	Sample to tag. shape=(n_samples, )
	:return:
	'''
	eps = np.random.normal(mu, sig, X.shape[0])
	return (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2) + eps

def Q10a():
	'''
	:return:	D_X - shape=(n_train_samp, )
				D_y - shape=(n_train_samp, )
				T_X - shape=(n_test_samp, )
				T_y - shape=(n_test_samp, )
	'''
	gc.enter_func()
	low, high = -3.2, 2.2
	m = 1500
	X = np.random.uniform(low, high, m)
	y = get_true_y(X)
	D_X, T_X = train_test_split(X, train_size=(2 / 3), shuffle=False)
	D_y, T_y = train_test_split(y, train_size=(2 / 3), shuffle=False)
	return D_X, D_y, T_X, T_y

def fit_pol(X, y, d):
	'''
	:param X:	shape=(n_samples, n_features)
	:param y: 	shape=(n_samples)
	:param d: 	polynomial degree
	:return:
	'''
	coeff = np.polyfit(X, y, d)
	return np.poly1d(coeff)

def Q10b(D_X, D_y):
	gc.enter_func()
	d = 15  # Pol deg
	S_X, V_X = train_test_split(D_X, train_size=0.5, shuffle=True)
	S_y, V_y = train_test_split(D_y, train_size=0.5, shuffle=True)
	h = [None] * d  # Hypotheses
	for deg in range(d):
		h[deg] = fit_pol(S_X, S_y, deg)
	return V_X, V_y, h

def mse(y, y_hat):
	'''
	:param y:		shape=(n_samples,)
	:param y_hat: 	shape=(n_samples,)
	:return: 		Mean Squared Error
	'''
	err = y - y_hat
	squared = err ** 2
	mean = np.mean(squared)
	return mean

def Q10c(V_X, V_y, h):
	'''
	:param V_X: 	Validation X. shape=(n_samples, n_features)
	:param V_y: 	Validation y. shape=(n_samples,)
	:param h: 		Hypotheses list. shape=[deg]
	:return:
	'''
	gc.enter_func()
	losses = np.zeros(len(h))
	for d, hyp in enumerate(h):
		y_hat = hyp(V_X)
		losses[d] = mse(V_y, y_hat)
	best_d = np.argmax(losses)[0]
	return h[best_d]

def Q10d(S_X, S_y):
	'''
	:param S_X:		shape=(n_samples, )
	:param S_y: 	shape=(n_samples, )
	:return:
	'''
	gc.enter_func()
	k = 5  # k-folds factor
	max_d = 15  # Max polynomial degree
	F_X, F_y = np.split(S_X, k), np.split(S_y, k)
	err_per_d = np.zeros(max_d)  # Array of errors per d
	# Run over all poly degrees
	for d in range(max_d):
		err_per_i = np.zeros(k)  # Array of errors per i
		# Run over all k folds
		for i in range(k):
			# np.random.shuffle(S_X)
			# p = np.random.permutation(S_X.shape[0])
			# S_X, S_y = S_X[p], S_y[p]
			# Create a list of data without the i-th fold
			F_X_d_l = [x for j, x in enumerate(F_X) if j != i]
			F_y_d_l = [y for j, y in enumerate(F_y) if j != i]
			# Concat them into an array
			F_X_d, F_y_d = np.concatenate(F_X_d_l), np.concatenate(F_y_d_l)
			# Put the k-d learner in its place
			poly = PolyFit(F_X_d, F_y_d, d)
			y_hat = poly.predict(F_X[i])

			err_per_i[i] = mse(F_y[i], y_hat)
		err_per_d[d] = np.mean(err_per_i)
	d_star = np.argmin(err_per_d)
	return err_per_d, d_star

def Q10e(err_per_d, d_star):
	gc.enter_func()
	gc.init_plt()
	plt.plot(err_per_d)
	gc.save_plt()
	gc.log(d_star)

def Qhelper(D_X, D_y):
	gc.init_plt()
	# plt.scatter(D_X, D_y)
	X = np.arange(-3, 3, 0.2)
	y = get_true_y(X)
	plt.scatter(X,y)
	gc.save_plt()

def Q10():
	gc.enter_func()
	D_X, D_y, T_X, T_y = Q10a()
	Qhelper(D_X, D_y)
	V_X, V_y, h = Q10b(D_X, D_y)
	# h_star = Q10c(V_X, V_y, h)
	err_per_d, d_star = Q10d(D_X, D_y)
	Q10e(err_per_d, d_star)
