import tensorflow as tf
import numpy as np
import scipy.optimize as opt
import scipy.sparse as spsp
from tqdm import tqdm


class MultiObjectiveProblem:
    def __init__(self, function, jacobian=None, dtype='float64'):
        """
        :param function: vectorial tf.function that for each N-by-n array of N n-dimensional vectors, returns
            an N-by-m matrix of N m-dimensional vectors
        :param jacobian: tensorial tf.function that for each N-by-n array of N n-dimensional vectors, returns
            an N-by-m-by-n tensor of N m-by-n Jacobians. If None (default), it is authomatically computed with
            Automatic Differentiation (if possible).
        :param dtype: data type (default float64)
        """
        self._function = function
        self.__input_jacobian = jacobian
        self._dtype = dtype

    def __call__(self, X):
        X = tf.cast(X, dtype=getattr(tf, self._dtype))
        Y = self._function(X)

        return Y

    def _jacobian(self, X):
        X = tf.cast(X, dtype=getattr(tf, self._dtype))
        if self.__input_jacobian is None:
            with tf.GradientTape() as g:
                g.watch(X)
                Y = self.__call__(X)

            J = g.batch_jacobian(Y, X)
        else:
            J = self.__input_jacobian(X)

        return J

    def __linprog_elements(self, X):
        J = self._jacobian(X)
        G_list = list(J.numpy())
        g_list = [G.sum(axis=0) for G in G_list]

        N, m, n = J.shape

        return G_list, g_list, N, m, n

    def __fliegesvaiter_linprog_elements(self, G_list_i, g_list_i, m, n, **kwargs):
        b_ub = np.zeros((m, 1))
        bounds = [(-1., 1.)] * n + [(None, None)]

        A_ub = G_list_i
        A_ub = np.hstack([A_ub, -np.ones((A_ub.shape[0], 1))])

        c = np.zeros_like(g_list_i)
        c = np.concatenate([c, [1.]])

        return c, A_ub, b_ub, bounds

    def __dellasanta_linprog_elements(self, G_list_i, g_list_i, m, n, **kwargs):
        Gi_norms = kwargs['Gi_norms']
        cbeta_add = kwargs['cbeta_add']

        gamma_G = np.abs(G_list_i).max()
        gamma_g = np.abs(g_list_i).max()
        gamma = np.max([gamma_g, gamma_G])

        b_ub = np.zeros((m, 1))
        bounds = [(-gamma, gamma)] * n + [(None, 0)]

        c = g_list_i

        Gi_normalized = G_list_i / Gi_norms

        A_ub = Gi_normalized
        A_ub = np.hstack([A_ub, -np.ones((A_ub.shape[0], 1))])

        beta_coeff = np.linalg.norm(c) + cbeta_add
        c = np.concatenate([c, [beta_coeff]])

        return c, A_ub, b_ub, bounds

    def _mgdd(self,
              X,
              linprog_method='highs',
              tolgrad_prods=1e-7,
              stop_list_previous=None,
              shared_descent_method='dellasanta',
              cbeta_add=1.
              ):

        G_list, g_list, N, m, n = self.__linprog_elements(X)

        if stop_list_previous is None:
            stop_list_previous = [False] * N

        res_list = []
        stop_list = []
        for i in range(N):
            if stop_list_previous[i]:
                res_list.append({'x': np.zeros(n + 1)})
                stop_list.append(True)
                continue  # SKIP NEXT OPERATIONS IN THE for CYCLE

            Gi_norms = np.linalg.norm(G_list[i], axis=1).reshape(m, 1)

            one_nullgrad_atleast = np.any(Gi_norms <= (tolgrad_prods ** 2))
            stop_list.append(one_nullgrad_atleast)

            if stop_list[i]:
                # print('STOP grads')
                res_list.append({'x': np.zeros(n + 1)})
                continue  # SKIP NEXT OPERATIONS IN THE for CYCLE

            # IN ORDER TO AVOID ZERO-DIVISION OPERATIONS
            Gi_norms[Gi_norms == 0] = tolgrad_prods ** 2

            c, A_ub, b_ub, bounds = getattr(self,
                                            f'_{self.__class__.__name__}__{shared_descent_method}_linprog_elements'
                                            )(
                G_list_i=G_list[i],
                g_list_i=g_list[i],
                m=m,
                n=n,
                Gi_norms=Gi_norms,
                cbeta_add=cbeta_add
            )

            res_i = opt.linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method=linprog_method
            )

            if res_i.x is None:
                res_i.x = np.zeros(n + 1)

            res_list.append(res_i)

        p_list = [res['x'][:-1] for res in res_list]

        G_times_p_list = [G_list[i] @ np.reshape(p_list[i], (n, 1)) for i in range(N)]

        return p_list, G_list, g_list, G_times_p_list, stop_list

    def _mgdd_blockwise(
            self,
            X,
            linprog_method='highs',
            tolgrad_prods=1e-7,
            stop_list_previous=None,
            shared_descent_method='dellasanta',
            cbeta_add=1.
            ):

        G_list, g_list, N, m, n = self.__linprog_elements(X)

        if stop_list_previous is None:
            stop_list_previous = [False] * N

        one_nullgrad_atleast = [np.any(np.linalg.norm(G_list[i], axis=1) <= (tolgrad_prods ** 2)) for i in range(N)]
        for i in np.argwhere(one_nullgrad_atleast).flatten():
            stop_list_previous[i] = True

        nonstop_inds = np.argwhere(np.logical_not(stop_list_previous)).flatten()
        # print(f'nonstop_inds: {nonstop_inds}')

        true_N = nonstop_inds.size
        stop_list = stop_list_previous.copy()

        if true_N > 0:
            c = []
            A_ub_blocks = []
            b_ub = np.empty((0, 1))
            bounds = []
            for i in nonstop_inds:
                Gi_norms = np.linalg.norm(G_list[i], axis=1).reshape(m, 1)

                one_nullgrad_atleast = np.any(Gi_norms <= (tolgrad_prods ** 2))
                stop_list[i] = one_nullgrad_atleast

                # IN ORDER TO AVOID ZERO-DIVISION OPERATIONS
                Gi_norms[Gi_norms == 0] = tolgrad_prods ** 2

                c_i, A_ub_i, b_ub_i, bounds_i = getattr(self,
                                                        f'_{self.__class__.__name__}__{shared_descent_method}_linprog_elements'
                                                        )(
                    G_list_i=G_list[i],
                    g_list_i=g_list[i],
                    m=m,
                    n=n,
                    Gi_norms=Gi_norms,
                    cbeta_add=cbeta_add
                )

                c = np.concatenate([c, c_i])
                # print(f'c_i: {c_i.shape}')
                # print(f'A_ub_i: {A_ub_i.shape}')
                A_ub_blocks.append(A_ub_i)

                b_ub = np.concatenate([b_ub, b_ub_i])
                bounds += bounds_i

            # print(f'len(A_ub_blocks): {len(A_ub_blocks)}')
            A_ub = spsp.block_diag(A_ub_blocks)

            # print('------------------------------')
            # print(f'c shape: {c.shape}')
            # print(f'A_ub shape: {A_ub.shape}')
            # print(f'b_ub shape: {b_ub.shape}')

            # print(f'c = {c}')
            # print(f'A_ub = {A_ub}')
            # print(f'b_ub = {b_ub}')
            # print(f'bounds = {bounds}')
            # print('------------------------------')

            res = opt.linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method=linprog_method
            )

            res_x_splitted = np.split(res.x, true_N)

        p_list = [np.zeros(n)] * N
        for i in range(true_N):
            ii = nonstop_inds[i]
            p_list[ii] = res_x_splitted[i][:-1]

        G_times_p_list = [G_list[i] @ np.reshape(p_list[i], (n, 1)) for i in range(N)]

        return p_list, G_list, g_list, G_times_p_list, stop_list

    def _which_nondominated(self, X, Y=None,
                            dominated_tol=1e-8, verbose=False, max_checks=None,
                            random_seed=None):

        if Y is None:
            Y = self(X).numpy()

        if random_seed is not None:
            np.random.seed(random_seed)

        N, m = Y.shape

        i_tocheck = list(range(N))
        i_nondominated = []
        i_dominated = set()

        if max_checks is None:
            max_checks = N
        else:
            max_checks = min(N, max_checks)

        if verbose:
            pbar_nondom = tqdm(total=max_checks)
            pbar_tocheck = tqdm(total=N)

        i = 0

        len_tocheck = len(i_tocheck)
        while len_tocheck > 0 and len(i_nondominated) < max_checks:

            i = i_tocheck.pop(np.random.choice(len_tocheck))

            yi = np.expand_dims(Y[i, :], axis=0)
            xi_is_nondominated = np.logical_or(
                np.any((yi - Y[i_tocheck + i_nondominated, :]) < 0., axis=1),
                (np.linalg.norm(Y[i_tocheck + i_nondominated, :] - yi, axis=1) <= dominated_tol).flatten()
            ).all()

            dominated_by_xi = np.argwhere(np.all(yi < Y[i_tocheck, :], axis=1)).flatten()
            dominated_by_xi = [i_tocheck[ii] for ii in dominated_by_xi]

            if xi_is_nondominated:
                i_nondominated.append(i)
            else:
                i_dominated = i_dominated.union({i})

            i_dominated = i_dominated.union(set(dominated_by_xi))
            i_tocheck = [ii for ii in i_tocheck if ii not in i_dominated]

            len_tocheck_old = len_tocheck
            len_tocheck = len(i_tocheck)

            if verbose:
                pbar_tocheck_update = len_tocheck_old - len_tocheck

            if verbose and xi_is_nondominated:
                pbar_nondom_update = 1
            else:
                pbar_nondom_update = 0

            if verbose:
                pbar_tocheck.update(pbar_tocheck_update)
                pbar_nondom.update(pbar_nondom_update)

        return i_nondominated

    def _which_nondominated_by(self, Yfront,
                               X, Y=None,
                               dominated_tol=1e-8, verbose=False
                               ):

        if Y is None:
            Y = self(X).numpy()

        N, m = Y.shape

        i_tocheck = list(range(N))
        i_nondominated = []
        i_dominated = set()
        front_dominated = {}

        if verbose:
            pbar = tqdm(total=N)
        else:
            pbar = range(N)

        len_tocheck = len(i_tocheck)
        while len_tocheck > 0:
            i = i_tocheck.pop(np.random.choice(len_tocheck))

            yi = np.expand_dims(Y[i, :], axis=0)
            xi_is_nondominated = np.logical_or(
                np.any((yi - Yfront) < 0., axis=1),
                (np.linalg.norm(Yfront - yi, axis=1) <= dominated_tol).flatten()
            ).all()

            if xi_is_nondominated:
                i_nondominated.append(i)

            front_dominated_by_xi = np.argwhere(np.all(yi < Yfront, axis=1)).flatten()
            if front_dominated_by_xi.size > 0:
                front_dominated[i] = front_dominated_by_xi

            dominated_by_xi = np.argwhere(np.all(yi < Y[i_tocheck, :], axis=1)).flatten()
            dominated_by_xi = [i_tocheck[ii] for ii in dominated_by_xi]

            i_dominated = i_dominated.union(set(dominated_by_xi))
            i_tocheck = [ii for ii in i_tocheck if ii not in i_dominated]

            len_tocheck_old = len_tocheck
            len_tocheck = len(i_tocheck)

            if verbose:
                pbar.update(len_tocheck_old - len_tocheck)

        return i_nondominated, front_dominated




