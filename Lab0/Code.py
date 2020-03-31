from minimization import *
import numpy as np

if __name__ == "__main__":
    Np = 4  # number of rows
    Nf = 4  # columns
    A = np.random.randn(Np, Nf)  # A Matrix
    y = np.random.randn(Np, 1)  # y column vector

    # LLS Algorithm
    m = SolveLLS(y, A)
    m.run()
    m.print_result('LLS')
    m.plot_w('LLS')

    # Gradient Algorithm
    Nit = 1000
    gamma = 1e-3
    g = SolveGrad(y, A)
    g.run(gamma, Nit)
    g.print_result('Gradient algo.')
    g.plot_err('Gradient algo: square error', logy=1, logx=1)

    # Steepest decent Algorithm
    Nit = 300
    gamma = 1e-3
    s = SteepestDec(y, A)
    s.run(gamma, Nit)
    s.print_result("Steepest decent")
    s.plot_err('Steepest decent: square error', logy=1, logx=0)

    # Stochastic Gradient Algorithm
    Nit = 10
    gamma = 1e-3
    sg = SolveStochGrad(y, A)
    sg.run(gamma, Nit)
    sg.print_result("Sthocastic Gradient")
    sg.plot_err("Stochastic Gradient", logy=1, logx=0)

    # Minibatches
    mini = SolveMinibatches(y, A)
    mini.run()
    mini.print_result('Minibatches Gradient')
    mini.plot_err('Minibatches Gradient', logy=1, logx=0)
