from Code.q7 import *

gc = Garcon()

def Q8corr(poscorr=True):
	'''
	:param poscorr:		Is the correlation positive?
	:return:
	'''
	gc.enter_func()
	# Eigen vectors
	V = (1.0 / np.sqrt(2)) * np.array([[1.0, 1], [1, -1]])
	D_corr = np.array([[2.0, 0], [0, 0.01]])
	D_ncorr = np.array([[0.01, 0], [0, 2]])
	# Diagonal mat
	D = D_corr if poscorr else D_ncorr
	# Covariance mat made using EVD
	cov = np.matmul(np.matmul(V, D), V.T)
	samp1, samp2 = Q7a(cov)
	qnum = '8a_' if poscorr else '8b_'
	Q7b(samp1, samp2, qnum)
	Q7c(samp1, samp2, qnum)

def Q8():
	gc.enter_func()
	Q8corr()
	Q8corr(False)
