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


def Q10a(sig):
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
	y = get_true_y(X, sig=sig)
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
	v_err_per_d, t_err_per_d = np.zeros(max_d), np.zeros(max_d)  # Array of (v)alidation and (t)rain errors per d
	# Run over all poly degrees
	for d in range(max_d):
		v_err_per_i, t_err_per_i = np.zeros(k), np.zeros(k)  # Array of errors per i
		# Run over all k folds
		for i in range(k):
			# Create a list of data without the i-th fold
			F_X_d_l = [x for j, x in enumerate(F_X) if j != i]
			F_y_d_l = [y for j, y in enumerate(F_y) if j != i]
			# Concat them into an array
			F_X_d, F_y_d = np.concatenate(F_X_d_l), np.concatenate(F_y_d_l)
			# Put the k-d learner in its place
			poly = PolyFit(F_X_d, F_y_d, d)
			y_hat_valid = poly.predict(F_X[i])
			y_hat_train = poly.predict(F_X_d)

			v_err_per_i[i], t_err_per_i[i] = mse(F_y[i], y_hat_valid), mse(F_y_d, y_hat_train)
		v_err_per_d[d] = np.mean(v_err_per_i)
		t_err_per_d[d] = np.mean(t_err_per_i)
	d_star = np.argmin(v_err_per_d)
	gc.log_var(d_star=d_star)
	return v_err_per_d, t_err_per_d, d_star


def Q10e(validation_errs, train_errs, d_star, sig):
	gc.enter_func()
	qnum = 'Q10e' if sig==1 else 'Q10h_e'
	gc.init_plt(f'{qnum}: Mean Errors, sig={sig}')
	plt.plot(validation_errs, label='Validation')
	plt.plot(train_errs, label='Train')
	plt.xlabel('d (degree of the polynomial)')
	plt.ylabel('Mean Err Over k-Folds')
	plt.legend()
	gc.save_plt(qnum)
	d_star_err = validation_errs[d_star]
	gc.log_var(d_star_err=d_star_err)


def Q10f(D_X, D_y, d_star):
	poly = PolyFit(D_X, D_y, d_star)
	return poly


def Q10g(T_X, T_y, poly):
	y_hat = poly.predict(T_X)
	err = mse(T_y, y_hat)
	gc.log_var(err=err)


def Q10sig(sig):
	gc.enter_func()
	D_X, D_y, T_X, T_y = Q10a(sig)
	# Qhelper(D_X, D_y)
	# V_X, V_y, h = Q10b(D_X, D_y)
	# h_star = Q10c(V_X, V_y, h)
	valid_err, train_err, d_star = Q10d(D_X, D_y)
	Q10e(valid_err, train_err, d_star, sig)
	poly = Q10f(D_X, D_y, d_star)
	Q10g(T_X, T_y, poly)

def Q10():
	gc.enter_func()
	Q10sig(1)
	Q10sig(5)
