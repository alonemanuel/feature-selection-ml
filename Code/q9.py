from Code.q7 import *

gc = Garcon()


def Q9():
	gc.enter_func()
	V = 0.5 * np.array([[np.sqrt(3), 1], [1, -np.sqrt(3)]])
	D = np.array([[2.0, 0], [0, 0.01]])
	cov = np.matmul(np.matmul(V, D), V.T)
	mu1, mu2 = np.array([0, 3 / 2]), np.array([0, -3 / 2])
	samp1, samp2 = Q7a(cov, mu1, mu2)
	qnum = '9a_'
	Q7b(samp1, samp2, qnum)
	Q7c(samp1, samp2, qnum)
