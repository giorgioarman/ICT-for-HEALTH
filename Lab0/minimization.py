import numpy as np
import matplotlib.pyplot as plt


class SolveMinProb:
    def __init__(self, y=np.ones((3, 1)), A=np.eye(3)):
        self.matr = A
        self.Np = y.shape[0]
        self.Nf = A.shape[1]
        self.vect = y
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def plot_w(self, title='Solution'):
        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w)
        plt.xlabel('n')
        plt.ylabel('w(n)')
        plt.title(title)
        plt.grid()
        plt.show()
        return

    def print_result(self, title):  # method to print
        print(title, ' ')
        print('the optimum weight vector is: ')
        print(self.sol)
        return

    def plot_err(self, title='Square_error', logy=0, logx=0):
        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)  # leave some space
        plt.grid()
        titleToSave = title.replace(' ', '').replace(':', '')
        plt.savefig("Images/" + titleToSave + ".png")
        plt.show()
        return


class SolveLLS(SolveMinProb):
    def run(self):
        A = self.matr
        y = self.vect
        w = np.dot(np.linalg.pinv(A), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w) - y)  # errore quadratico minimo trovato
        print("self min : ", self.min)


class SolveGrad(SolveMinProb):
    def run(self, gamma=1e-3, Nit=100):  # we need to specify the params
        self.err = np.zeros((Nit, 2), dtype=float)
        # the value of the function to be minimized: the first column
        # stores the iteration step, the second column stores the value of the error
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)  # random initialization of w
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]


class SteepestDec(SolveMinProb):
    def run(self, gamma=1e-3, Nit=1000):  # we need to specify the params
        self.err = np.zeros((Nit, 2), dtype=float)
        # the value of the function to be minimized: the first column
        # stores the iteration step, the second column stores the value of the error
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)  # random initialization of w
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            H = np.dot(A, A.T)
            gamma = np.power(np.linalg.norm(grad), 2) / np.dot(np.dot(grad.T, H), grad)
            w = w - (gamma / 4) * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]


class SolveStochGrad(SolveMinProb):  # CONTROLLA RIGHE E COLONNE
    def run(self, gamma=1e-3, Nit=100):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        Nf = A.shape[1]
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        it = 0
        j = 0
        for it in range(Nit):
            for j in range(Nf):
                grad = np.dot(np.dot(A.T[j], w) - y.T, A.T[j])  # A.T[j] accdo alla riga
                w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]


class ConjGrad(SolveMinProb):
    def run(self):
        self.err = np.zeros((self.Nf, 2), dtype=float)
        A = self.matr
        y = self.vect
        Q = np.dot(A.T, A)  # --------------- A,A.T
        Nf = A.shape[1]
        beta = 0
        w = np.zeros((self.Nf, 1), dtype=float)
        # grad = 2 * np.dot(A.T, (np.dot(A, w) - y)) # grad for w = 0
        b = np.dot(A.T, y)
        grad = -b
        d = -grad
        for it in range(self.Nf):
            # grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            alpha = - np.dot(d.T, grad) / np.dot(np.dot(d.T, Q), d)  # --------------------grad*d.T/np.dot(Q,d)*d.T
            w = w + d * alpha
            grad = grad + alpha * np.dot(Q, d)  # ----------------d,Q
            beta = np.dot(np.dot(grad.T, Q), d) / np.dot(d.T * Q, d)
            d = -grad + d * beta
            # gamma = np.power(np.linalg.norm(grad), 2) / np.dot(np.dot(grad.T, Q), grad)
            # w = w - (gamma / 4) * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]
        return


class SolveMinibatches(SolveMinProb):

    def run(self, gamma=1e-3, Nit=500, K=10):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        K_1 = int(self.Np / K)

        for it in range(Nit):
            k = 0
            for i in range(0, K_1):
                X = A[range(k, k + K_1), :]
                Y = y[range(k, k + K_1), :]
                grad_i = 2 * (np.dot(np.dot(X.T, X), w) - np.dot(X.T, Y))
                w = w - gamma * grad_i
                k = k + K_1
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]
