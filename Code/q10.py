import numpy as np
from Code.garcon import Garcon
from sklearn.model_selection import train_test_split

gc = Garcon()


def get_true_y(X, mu=0, sig=1):
	'''
	:param X:	Sample to tag. shape=(n_samples, )
	:return:
	'''
	eps = np.random.normal(mu, sig)
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
	D_X, T_X = train_test_split(X)
	D_y, T_y = train_test_split(y)
	return D_X, D_y, T_X, T_y

def Q10b(D_X, D_y):

	gc.enter_func()
	S_X, V_X = train_test_split(D_X)
	S_y, V_y = train_test_split(D_y)

def Q10():
	gc.enter_func()
	D_X, D_y, T_X, T_y = Q10a()
	Q10b(D_X, D_y)
