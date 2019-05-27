from Code.garcon import Garcon
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

gc = Garcon()


def Q7d(samp1, samp2):
    '''
    :param samp1: shape=(n_samples, 2)
    :param samp2: shape=(n_samples, 2)
    :return:
    '''
    gc.enter_func()
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    samp1_R = np.matmul(R, samp1.T).T
    samp2_R = np.matmul(R, samp2.T).T
    Q7b(samp1_R, samp2_R, qnum='7d_', rotated=True)
    Q7c(samp1_R, samp2_R, qnum='7d_', rotated=True)


def Q7c(samp1, samp2, qnum='7', rotated=False):
    '''
    :param samp1: shape=(n_samples, 2)
    :param samp2: shape=(n_samples, 2)
    :return:
    '''
    gc.enter_func()
    fn_pref = f'Q{qnum}c'
    rot_title = ', Rotated' if rotated else ''
    gc.init_plt()
    plt.subplot(2, 1, 1)
    plt.title(f'{fn_pref}: Feature 1{rot_title}')
    plt.hist(samp1[:, 0], density=True, edgecolor='k', alpha=0.5,
             label='Popu 1')
    plt.hist(samp2[:, 0], density=True, edgecolor='k', alpha=0.5,
             label='Popu 2')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f'{fn_pref}: Feature 2{rot_title}')
    plt.hist(samp1[:, 1], density=True, edgecolor='k', alpha=0.5,
             label='Popu 1')
    plt.hist(samp2[:, 1], density=True, edgecolor='k', alpha=0.5,
             label='Popu 2')
    plt.legend()
    gc.save_plt(f'{fn_pref}')


def Q7b(samp1, samp2, qnum='7', rotated=False):
    '''
    :param samp1: shape=(n_samples, 2)
    :param samp2: shape=(n_samples, 2)
    :return:
    '''
    gc.enter_func()
    fn_pref = f'Q{qnum}b'
    rot_title = ', Rotated' if rotated else ''
    gc.init_plt(f'{fn_pref}: Normal Distribution{rot_title}')
    plt.scatter(samp1[:, 0], samp1[:, 1], alpha=0.5, label='Population 1')
    plt.scatter(samp2[:, 0], samp2[:, 1], alpha=0.5, label='Population 2')
    plt.legend()
    gc.save_plt(f'{fn_pref}')


def Q7a(cov, mu1=None, mu2=None):
    gc.enter_func()
    m = 1000
    if (mu1 is None) and (mu2 is None):
        mu1, mu2 = np.array([1, 1]), np.array([-1, -1])
    else:
        mu1, mu2 = mu1, mu2
    samp1 = np.random.multivariate_normal(mu1, cov, m)
    samp2 = np.random.multivariate_normal(mu2, cov, m)
    return samp1, samp2


def Q7warmup():
    gc.enter_func()
    m = 1000
    dist = stats.norm()
    samp = dist.rvs(m)
    x = np.linspace(start=stats.norm.ppf(0.001), stop=stats.norm.ppf(0.999),
                    num=250)
    gc.init_plt('Q7warmup: PDF and Histogram, Normal Distribution')
    plt.plot(x, dist.pdf(x), label='PDF')
    plt.hist(samp, density=True, edgecolor='k', alpha=0.7, label='Histogram')
    plt.legend()
    gc.save_plt()


def Q7():
    Q7warmup()
    cov = np.eye(2)
    samp1, samp2 = Q7a(cov)
    Q7b(samp1, samp2)
    Q7c(samp1, samp2)
    Q7d(samp1, samp2)


def Q8corr(poscorr=True):
    gc.enter_func()
    V = (1.0 / np.sqrt(2)) * np.array([[1.0, 1], [1, -1]])
    D_corr = np.array([[2.0, 0], [0, 0.01]])
    D_ncorr = np.array([[0.01, 0], [0, 2]])
    D = D_corr if poscorr else D_ncorr
    cov = np.matmul(np.matmul(V, D), V.T)
    samp1, samp2 = Q7a(cov)
    qnum = '8a_' if poscorr else '8b_'
    Q7b(samp1, samp2, qnum)
    Q7c(samp1, samp2, qnum)


def Q8():
    gc.enter_func()
    Q8corr()
    Q8corr(False)


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


def main():
    Q7()
    Q8()
    Q9()


if __name__ == '__main__':
    main()
