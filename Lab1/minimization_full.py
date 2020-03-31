import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats


class SolveMinProb:
    def __init__(self, dir=None, A_train =None, A_test=None, A_val=None,
                 y_train=None, y_test=None, y_val=None, mean=None, st_dev=None,  y=np.ones((3,1)), A = np.eye(3),):
        '''
        :param A_test: matrix of test data set
        :param w: is the optimum weight vector
        :param A_train: matrix of train data set. This is used to find w
        :param y_test: column taken from the matrix A_test, denormalized
        :param y_train: column taken from the matrix A_train, denormalized
        :param y_hat_test:calculated with matrix at which each vector belongs times w, the weight vector. We actually know what the true y_test is and so we can measure the estimation error on the testing data e_test = y_test - y_hat_test
        and then we can calculate the mean square error for the testing data MSE_test = ||e_test||^2/N_test(rows)
        :param mean: row vector: each element is the mean calculated for each column of the  matrix containing all data
        :param st_dev: vector of mean calculated for each column of the  matrix containing all data
        '''

        self.matr = A
        self.vect = y
        self.Np = y_train.shape[0]  # Number of patients
        self.Nf = A_train.shape[1]  # Number of features
        self.sol = np.zeros((self.Nf, 1), dtype=float)  # Initialize sol

        self.dir = dir
        self.A_test= A_test
        self.A_train = A_train
        self.A_val = A_val
        self.y_train = y_train.reshape(self.Np, 1)
        self.y_test = y_test.reshape(len(y_test), 1)
        self.y_val = y_val.reshape(len(y_val), 1)
        self.mean = mean
        self.st_dev = st_dev
        return

    def plot_w(self, title='Values of w'):
        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.stem(n, w, use_line_collection=True)
        plt.xlabel('features')
        plt.ylabel('w for each feature')
        plt.ylim(ymin=-0.75, ymax=1)
        plt.title(title)
        plt.grid()
        titleToSave = title.replace(' ', '').replace(':', '')
        imageDir = "Images/" + self.dir + "/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        plt.savefig(imageDir + titleToSave + ".png")
        plt.show()
        return

    def plot_y(self, title='Value of Y'):

        w = self.sol
        '''Here denormalize the vector y'''
        y_hat_train = np.dot(self.A_train, w)*self.st_dev + self.mean
        y_hat_test = np.dot(self.A_test, w)*self.st_dev + self.mean
        y_train = self.y_train*self.st_dev + self.mean
        y_test = self.y_test*self.st_dev + self.mean

        plt.title(title + " (Training Set)")
        plt.figure()
        plt.scatter(y_train, y_hat_train, s=3)
        lined = [min(y_train), max(y_train)]
        plt.plot(lined, lined, color='black')
        plt.ylabel('ŷ_train')
        plt.xlabel('y_train')
        plt.title(title +" (Train)")
        plt.grid()
        titleToSave = title.replace(' ', '').replace(':', '')
        imageDir = "Images/" + self.dir + "/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        plt.savefig(imageDir + titleToSave + "_train.png")
        plt.show()

        plt.title(title + " (Test Set)")
        plt.figure()
        plt.scatter(y_test, y_hat_test, s=3, color='orange')
        plt.plot(y_test, y_test, color='black')
        plt.ylabel('ŷ_test')
        plt.xlabel('y_test')
        plt.title(title + " (Test)")
        titleToSave = title.replace(' ', '').replace(':', '')
        imageDir = "Images/" + self.dir + "/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        plt.savefig(imageDir + titleToSave + "_test.png")
        plt.title(title)
        plt.grid()
        plt.show()

    def plot_hist(self, title='Histogram'):
        print(title, '')
        '''
        This method is used to plot the istograms of y_hat_train-y_hat and y_hat_test-y_test
        '''

        w = self.sol
        y_hat_train = np.dot(self.A_train, w) * self.st_dev + self.mean
        y_hat_test = np.dot(self.A_test, w) * self.st_dev + self.mean
        y_train = self.y_train * self.st_dev + self.mean
        y_test = self.y_test * self.st_dev + self.mean

        error_test = y_test - y_hat_test
        error_train = y_train - y_hat_train

        plt.hist(error_train, bins=50, alpha=0.7, color='blue', label='Train Set')
        plt.hist(error_test, bins=50, alpha=0.7, color='orange', label='Test Set')
        plt.xlabel('(Estimated Y - Real Y)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid()
        plt.xlim(xmin=-10, xmax=15)
        plt.ylim(ymin=0, ymax=200)
        plt.legend(loc='upper right')
        titleToSave = title.replace(' ', '').replace(':', '')
        imageDir = "Images/" + self.dir + "/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        plt.savefig(imageDir + titleToSave + "_test.png", bbox_inches='tight')
        plt.show()
        return

    def print_result(self, title='Result'):  # method to print
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
        imageDir = "Images/" + self.dir + "/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        plt.savefig(imageDir + titleToSave + ".png")
        plt.show()
        return

class SolveLLS (SolveMinProb):
    def run(self):

        w = np.dot(np.linalg.pinv(self.A_train), self.y_train)
        # w = np.dot(np.dot(np.linalg.inv(np.dot(self.A_train.T, self.A_train)), self.A_train.T), self.y_train)
        self.sol = w
        self.MSE_train = np.linalg.norm((np.dot(self.A_train, w)*self.st_dev+self.mean) -
                                        (self.y_train*self.st_dev+self.mean))**2/self.A_train.shape[0]
        self.MSE_test = np.linalg.norm((np.dot(self.A_test, w)*self.st_dev+self.mean) -
                                       (self.y_test*self.st_dev+self.mean))**2/self.A_test.shape[0]
        self.MSE_val = np.linalg.norm((np.dot(self.A_val, w)*self.st_dev+self.mean) -
                                      (self.y_val*self.st_dev+self.mean))**2/self.A_val.shape[0]
        print("MSE of Train")
        print(self.MSE_train)
        print("MSE of test")
        print(self.MSE_test)
        print("MSE of val")
        print(self.MSE_val)

'''
For the iterative algorithms in order to evaluate the MSE it has been calculated in each 
iteration error_val (as y_val - y_hat_val), error_train (as y_train - y_hat_train) 
and error_test (as y_test - y_hat_test) and a matrix self.err has been uploaded with this values.
'''
class SolveRidge(SolveMinProb):
    """" Ridge Algorithm """
    def run(self):
        np.random.seed(3)
        # w = np.zeros
        w = np.random.rand(self.Nf, 1)
        I = np.eye(self.Nf)
        Nit = 300
        self.err = np.zeros((Nit, 4), dtype=float)
        for it in range(Nit):
            w = np.dot(np.dot(np.linalg.inv(np.dot(self.A_train.T, self.A_train)+float(it)*I), self.A_train.T), self.y_train)
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm((np.dot(self.A_train, w)*self.st_dev +self.mean) - (self.y_train*self.st_dev +self.mean))**2 / self.A_train.shape[0]
            self.err[it, 2] = np.linalg.norm((np.dot(self.A_val, w)*self.st_dev +self.mean) - (self.y_val*self.st_dev +self.mean))**2 / self.A_val.shape[0]
            self.err[it, 3] = np.linalg.norm((np.dot(self.A_test, w)*self.st_dev +self.mean) - (self.y_test*self.st_dev +self.mean)) ** 2 / self.A_test.shape[0]
        best_lamb = np.argmin(self.err[:, 2])
        w = np.dot(np.dot(np.linalg.inv(np.dot(self.A_train.T, self.A_train) + best_lamb * I), self.A_train.T), self.y_train)
        print("MSE of Train")
        print(min(self.err[:, 1]))
        print("MSE of test")
        print(min(self.err[:, 3]))
        print("MSE of val")
        print(min(self.err[:, 2]))
        self.sol = w
        err = self.err
        print("best lambda is :", best_lamb)
        plt.figure()
        plt.plot(err[:, 0], err[:, 1], label='train')
        plt.plot(err[:, 0], err[:, 2], label='val')
        plt.xlabel('lambda')
        plt.ylabel('Mean Square Error')
        plt.legend()
        plt.title('Ridge error respect to lambda')
        plt.margins(0.01, 0.1)
        plt.xlim(xmin=0, xmax=300)
        plt.grid()
        plt.show()


class SolveGrad(SolveMinProb):
    def run(self, gamma = 1e-3, Nit = 200): #we need to specify the params
        self.err = np.zeros((Nit,4), dtype=float)
        '''
        :param gamma: learning coefficient. It's better to start 
        with small value of gamma and gradually manually increase it, 
        otherwise the algorithm could not converge. The correct value of 
        gamma depends on the specific func
        '''
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad = 2 * np.dot(self.A_train.T,(np.dot(self.A_train, w)-self.y_train))
            w = w - gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm((np.dot(self.A_train, w) * self.st_dev + self.mean) -
                                             (self.y_train * self.st_dev + self.mean)) ** 2 / self.A_train.shape[0]
            self.err[it, 3] = np.linalg.norm((np.dot(self.A_test, w) * self.st_dev + self.mean) -
                                             (self.y_test * self.st_dev + self.mean)) ** 2 / self.A_test.shape[0]
            self.err[it, 2] = np.linalg.norm((np.dot(self.A_val, w) * self.st_dev + self.mean) -
                                             (self.y_val * self.st_dev + self.mean)) ** 2 / self.A_val.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        self.min = [min(self.err[:, 1]), min(self.err[:, 2]), min(self.err[:, 3])]


class SolveStochGrad(SolveMinProb):
    def run(self, gamma=1e-3, Nit=100):
        self.err = np.zeros((Nit, 4), dtype=float)
        Nf=self.A_train.shape[1]
        Np=self.A_train.shape[0]
        np.random.seed(3)
        w = np.random.rand(self.Nf, 1)
        row = np.zeros((1,Nf), dtype = float)
        for it in range(Nit):
            for i in range(Np):
                for j in range(Nf):
                    row[0,j] = self.A_train[i,j]
                grad = 2*row.T* (np.dot(row, w)-self.y_train[i])
                w = w-gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm((np.dot(self.A_train, w) * self.st_dev + self.mean) -
                                             (self.y_train * self.st_dev + self.mean)) ** 2 / self.A_train.shape[0]
            self.err[it, 3] = np.linalg.norm((np.dot(self.A_test, w) * self.st_dev + self.mean) -
                                             (self.y_test * self.st_dev + self.mean)) ** 2 / self.A_test.shape[0]
            self.err[it, 2] = np.linalg.norm((np.dot(self.A_val, w) * self.st_dev + self.mean) -
                                             (self.y_val * self.st_dev + self.mean)) ** 2 / self.A_val.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        self.min = [min(self.err[:, 1]), min(self.err[:, 2]), min(self.err[:, 3])]


class SolveSteepestDec(SolveMinProb):
    def run(self, gamma = 1e-3, Nit = 100):
        self.err = np.zeros((Nit,4), dtype=float)
        w = np.random.rand(self.Nf, 1)
        '''
        :param gamma: the learning coefficient; it has to be optimized. 
        It's no more settled manually as in the gradient algorithm
        '''
        for it in range(Nit):
            grad = 2*np.dot(self.A_train.T, (np.dot(self.A_train, w)-self.y_train))
            H = 2*np.dot(self.A_train.T, self.A_train)
            gamma = np.power(np.linalg.norm(grad), 2) / np.dot(np.dot(grad.T, H), grad)
            w = w - gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm((np.dot(self.A_train, w) * self.st_dev + self.mean) -
                                             (self.y_train * self.st_dev + self.mean)) ** 2 / self.A_train.shape[0]
            self.err[it, 3] = np.linalg.norm((np.dot(self.A_test, w) * self.st_dev + self.mean) -
                                             (self.y_test * self.st_dev + self.mean)) ** 2 / self.A_test.shape[0]
            self.err[it, 2] = np.linalg.norm((np.dot(self.A_val, w) * self.st_dev + self.mean) -
                                             (self.y_val * self.st_dev + self.mean)) ** 2 / self.A_val.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        self.min = [min(self.err[:, 1]), min(self.err[:, 2]), min(self.err[:, 3])]


class SolveConjGrad(SolveMinProb):
    def run(self):
        self.err = np.zeros((self.Nf, 4), dtype=float)
        ww = np.zeros((self.Nf, 1), dtype=float)
        Q = 2*np.dot(self.A_train.T, self.A_train)
        b = np.dot(self.A_train.T, self.y_train)
        grad = -b
        d = -grad
        for it in range(self.A_train.shape[1]):
            alpha = - (np.dot(d.T, grad)/np.dot(np.dot(d.T,Q),d))
            ww = ww + alpha*d
            grad = grad + alpha*np.dot(Q,d)
            beta = (np.dot(np.dot(grad.T,Q),d)/np.dot(np.dot(d.T,Q),d))
            d = -grad + d*beta
            self.err[it, 0] = it
            self.err[it, 1] = (np.linalg.norm((np.dot(self.A_train, ww)*self.st_dev + self.mean) - ((self.y_train*self.st_dev) + self.mean)))**2/len(self.y_train)
            self.err[it, 3] = (np.linalg.norm((np.dot(self.A_test, ww)*self.st_dev + self.mean) - ((self.y_test*self.st_dev) + self.mean)))**2/len(self.y_test)
            self.err[it, 2] = (np.linalg.norm((np.dot(self.A_val, ww)*self.st_dev + self.mean) - ((self.y_val*self.st_dev) + self.mean)))**2/len(self.y_val)
        print("MSE of Train")
        print(min(self.err[:, 1]))
        print("MSE of test")
        print(min(self.err[:, 3]))
        print("MSE of val")
        print(min(self.err[:, 2]))
        self.sol = ww
        self.min = [min(self.err[:, 1]), min(self.err[:, 2]), min(self.err[:, 3])]

