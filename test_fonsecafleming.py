from multi_objective_problems.test_problems import fonsecafleming as problem
from multiple_gradient_descent.algorithms import BacktrackingMultipleGradientDescent
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


N = 100
n = 3

X0 = -2 + 4 * np.random.rand(N, n)
Y0 = problem(X0).numpy()

bttype = 'dellasanta'  # admitted values: 'dellasanta', 'dellasanta_nostore', 'strictly_decreasing'
kmax = 250

shared_descent_method = 'dellasanta'  # admitted values: 'dellasanta', 'fliegesvaiter'
blockwise = True


mgd = BacktrackingMultipleGradientDescent(bttype=bttype,
                                          steplen=1., kmax=kmax, btfactor=0.8, btmax=40,
                                          c1=1e-9
                                          )

Xnew, Ynew, history = mgd.minimize(X0, problem,
                                   store_steps=False,
                                   verbose=True,
                                   linprog_method='highs',
                                   tolgrad_prods=1e-7,
                                   tolsteps=0.,
                                   shared_descent_method=shared_descent_method,
                                   cbeta_add=1.,
                                   dominated_tol=1e-8,
                                   elwise_equality_tol=1e-7,
                                   blockwise=blockwise
                                   )

if history['X_leftback_merged'] is not None:
    Xfin = np.vstack([Xnew, history['X_leftback_merged']])
    Yfin = np.vstack([Ynew, history['Y_leftback_merged']])
else:
    Xfin = Xnew
    Yfin = Ynew

fig1 = plt.figure()
plt.scatter(Yfin[:, 0], Yfin[:, 1], c='red', marker='+', s=6)
plt.scatter(history['Y_final_nondominated'][:, 0], history['Y_final_nondominated'][:, 1], c='blue', marker='x', s=10)
plt.title("Fonseca-Fleming - Objectives' Space")
plt.xlabel('f1')
plt.ylabel('f2')

fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
ax.scatter(Xfin[:, 0], Xfin[:, 1], Xfin[:, 2], c='red', marker='+', s=6)
ax.scatter(history['X_final_nondominated'][:, 0],
           history['X_final_nondominated'][:, 1],
           history['X_final_nondominated'][:, 2],
           c='blue', marker='x', s=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title("Fonseca-Fleming - Domain")

plt.show()


