from tqdm import tqdm
import numpy as np


class BacktrackingMultipleGradientDescent:
    def __init__(self, bttype='dellasanta', steplen=1., kmax=250,
                 btfactor=0.8, btmax=50, c1=1e-4,
                 ):
        self._bttype = bttype
        self._steplen = steplen
        self._kmax = kmax
        self._btfactor = btfactor
        self._btmax = btmax
        self._c1 = c1
    
    def __armijo_cycle(self, moo_problem, Xnew, Ynew, p_list, G_times_p_list, steplen_array):
        steplen_array_ = steplen_array.copy()
        Xnew_ = Xnew + steplen_array_ * np.vstack(p_list)

        Yarmijo = Ynew + self._c1 * steplen_array_ * np.hstack(G_times_p_list).T

        armijo_check = moo_problem(Xnew_).numpy() <= Yarmijo

        armijo_not_satisfied = np.logical_not(np.all(armijo_check, axis=1))

        bt = 0
        while np.any(armijo_not_satisfied) and bt < self._btmax:
            bt += 1

            steplen_array_[armijo_not_satisfied] = steplen_array_[armijo_not_satisfied] * self._btfactor

            Xnew_ = Xnew + steplen_array_ * np.vstack(p_list)

            Yarmijo = Ynew + self._c1 * steplen_array_ * np.hstack(G_times_p_list).T

            armijo_check = moo_problem(Xnew_).numpy() <= Yarmijo

            armijo_not_satisfied = np.logical_not(np.all(armijo_check, axis=1))
        
        return Xnew_, steplen_array_, armijo_not_satisfied, bt

    def __dellasanta_nostore_backtracking(self,
                                          moo_problem,
                                          Xnew, Ynew, p_list, G_times_p_list, steplen_array,
                                          dominated_tol, tolsteps,
                                          stop_list,
                                          X_leftback, Y_leftback
                                          ):

        Xnew_, steplen_array_, armijo_not_satisfied, bt = self.__armijo_cycle(
            moo_problem,
            Xnew, Ynew, p_list, G_times_p_list, steplen_array
        )

        new_pts_nondominated = np.logical_or(
            np.any((moo_problem(Xnew_).numpy() - Ynew) < 0., axis=1),
            (np.linalg.norm(Ynew - moo_problem(Xnew_).numpy(), axis=1) <= dominated_tol).flatten()
        )
        armijo_not_satisfied_and_dominated = np.logical_and(
            armijo_not_satisfied,
            np.logical_not(new_pts_nondominated)
        )

        if bt == self._btmax and np.any(armijo_not_satisfied_and_dominated):
            steplen_array_[armijo_not_satisfied_and_dominated] = 0.

        stepmat = steplen_array_ * np.vstack(p_list)
        step_norms = np.linalg.norm(stepmat, axis=1)
        where_toosmall_steps = np.argwhere(step_norms <= tolsteps).flatten()
        for ii in where_toosmall_steps:
            stop_list[ii] = True

        return Xnew, stepmat, steplen_array_, stop_list, X_leftback, Y_leftback
        
    def __dellasanta_backtracking(self, 
                                  moo_problem, 
                                  Xnew, Ynew, p_list, G_times_p_list, steplen_array, 
                                  dominated_tol, tolsteps,
                                  stop_list,
                                  X_leftback, Y_leftback
                                  ):
        
        Xnew_, steplen_array_, armijo_not_satisfied, bt = self.__armijo_cycle(
            moo_problem, 
            Xnew, Ynew, p_list, G_times_p_list, steplen_array
        )
        
        new_pts_nondominated = np.logical_or(
            np.any((moo_problem(Xnew_).numpy() - Ynew) < 0., axis=1),
            (np.linalg.norm(Ynew - moo_problem(Xnew_).numpy(), axis=1) <= dominated_tol).flatten()
        )
        armijo_not_satisfied_and_dominated = np.logical_and(
            armijo_not_satisfied,
            np.logical_not(new_pts_nondominated)
        )

        if bt == self._btmax and np.any(armijo_not_satisfied_and_dominated):
            steplen_array_[armijo_not_satisfied_and_dominated] = 0.

        stepmat = steplen_array_ * np.vstack(p_list)
        step_norms = np.linalg.norm(stepmat, axis=1)
        where_toosmall_steps = np.argwhere(step_norms <= tolsteps).flatten()
        for ii in where_toosmall_steps:
            stop_list[ii] = True

        # ------ PART FOR STORING NON-DOM. PTS BUT WITH NON-ZERO STEPS -------------
        pts_nondominated = np.logical_or(
            np.any((Ynew - moo_problem(Xnew + stepmat).numpy()) < 0., axis=1),
            (np.linalg.norm(Ynew - moo_problem(Xnew + stepmat).numpy(), axis=1) <= dominated_tol).flatten()
        )

        mask_leftback = np.logical_and(pts_nondominated, new_pts_nondominated)
        # print(np.argwhere(mask_leftback).flatten())
        for jj in np.argwhere(mask_leftback).flatten():
            X_leftback[jj].append(np.expand_dims(Xnew[jj, :], axis=0))
            Y_leftback[jj].append(np.expand_dims(Ynew[jj, :], axis=0))

        # ------------------------------------------------------------------------------
        
        return Xnew, stepmat, steplen_array_, stop_list, X_leftback, Y_leftback

    def __strictly_decreasing_backtracking(self, 
                                           moo_problem, 
                                           Xnew, Ynew, p_list, G_times_p_list, steplen_array, 
                                           dominated_tol, tolsteps,
                                           stop_list,
                                           X_leftback, Y_leftback
                                           ):
        Xnew_, steplen_array_, armijo_not_satisfied, bt = self.__armijo_cycle(
            moo_problem,
            Xnew, Ynew, p_list, G_times_p_list, steplen_array
        )

        if bt == self._btmax and np.any(armijo_not_satisfied):
            steplen_array_[armijo_not_satisfied] = 0.

        stepmat = steplen_array_ * np.vstack(p_list)
        step_norms = np.linalg.norm(stepmat, axis=1)
        where_toosmall_steps = np.argwhere(step_norms <= tolsteps).flatten()
        for ii in where_toosmall_steps:
            stop_list[ii] = True

        return Xnew, stepmat, steplen_array_, stop_list, X_leftback, Y_leftback

    def minimize(self,
                 X0, moo_problem,
                 store_steps=False,
                 verbose=True,
                 linprog_method='highs',
                 tolgrad_prods=1e-7,
                 tolsteps=0.,
                 shared_descent_method='dellasanta',
                 cbeta_add=1.,
                 dominated_tol=1e-8,
                 elwise_equality_tol=1e-7,
                 blockwise=True
                 ):
        if store_steps:
            X_list = [X0]
            Y_list = [moo_problem(X0).numpy()]
            allp_list = []
            allG_list = []
            allG_p_list = [None]
            allstop_list = [[False] * X0.shape[0]]
            allstep_list = []
        else:
            X_list = None
            Y_list = None
            allp_list = None
            allG_list = None
            allG_p_list = None
            allstop_list = [[False] * X0.shape[0]]
            allstep_list = None

        Xnew = X0
        Ynew = moo_problem(Xnew).numpy()

        # ------ PART FOR STORING NON-DOM. PTS BUT WITH NON-ZERO STEPS -------------
        X_leftback = [[] for i in range(X0.shape[0])]
        Y_leftback = [[] for i in range(X0.shape[0])]
        # ------------------------------------------------------------------------------

        steplen_array = self._steplen * np.ones((X0.shape[0], 1))

        if verbose:
            range_kmax = tqdm(range(self._kmax))
        else:
            range_kmax = range(self._kmax)

        for k in range_kmax:
            if not blockwise:
                p_list, G_list, g_list, G_times_p_list, stop_list = moo_problem._mgdd(
                    Xnew,
                    linprog_method=linprog_method,
                    tolgrad_prods=tolgrad_prods,
                    shared_descent_method=shared_descent_method,
                    cbeta_add=cbeta_add,
                    stop_list_previous=allstop_list[-1],
                )
            else:
                p_list, G_list, g_list, G_times_p_list, stop_list = moo_problem._mgdd_blockwise(
                    Xnew,
                    linprog_method=linprog_method,
                    tolgrad_prods=tolgrad_prods,
                    shared_descent_method=shared_descent_method,
                    cbeta_add=cbeta_add,
                    stop_list_previous=allstop_list[-1],
                )

            Xnew, stepmat, steplen_array_, stop_list, X_leftback, Y_leftback = (
                getattr(self,
                        f'_{self.__class__.__name__}__{self._bttype}_backtracking'
                        )(
                    moo_problem,
                    Xnew, Ynew, p_list, G_times_p_list, steplen_array,
                    dominated_tol, tolsteps,
                    stop_list,
                    X_leftback, Y_leftback
                )
            )

            # STOPPING CRITERIA VERIFIED FOR ALL THE POINTS
            if np.linalg.norm(stepmat, axis=1).max() <= tolsteps:
                break

            Xnew = Xnew + stepmat
            Ynew = moo_problem(Xnew).numpy()

            if store_steps:
                X_list.append(Xnew)
                Y_list.append(Ynew)
                allp_list.append(p_list)
                allG_list.append(G_list)
                allG_p_list.append(G_times_p_list)
                allstep_list.append(steplen_array_)
                allstop_list.append(stop_list)
            else:
                allstop_list = [stop_list.copy()]

        if store_steps:
            J = moo_problem._jacobian(Xnew)
            allG_list.append(list(J.numpy()))
            allp_list.append(None)
            allstop_list.append(None)
            allstep_list.append(None)

        Ynew = moo_problem(Xnew).numpy()

        if self._bttype == 'dellasanta':
            try:
                for i in range(X0.shape[0]):
                    if len(X_leftback[i]) > 0:
                        X_leftback_i = np.vstack(X_leftback[i])
                        Y_leftback_i = np.vstack(Y_leftback[i])

                        Y_leftback_i, unique_inds = np.unique(
                            np.round(Y_leftback_i, decimals=int(-np.log10(elwise_equality_tol))),
                            axis=0, return_index=True
                        )
                        X_leftback_i = X_leftback_i[unique_inds, :]

                        X_leftback[i] = X_leftback_i
                        Y_leftback[i] = Y_leftback_i

                X_leftback_merged = np.vstack([Xlb for Xlb in X_leftback if len(Xlb) > 0])
                Y_leftback_merged = np.vstack([Ylb for Ylb in Y_leftback if len(Ylb) > 0])

                Y_leftback_merged, unique_inds = np.unique(
                    np.round(Y_leftback_merged, decimals=int(-np.log10(elwise_equality_tol))),
                    axis=0, return_index=True
                )
                X_leftback_merged = X_leftback_merged[unique_inds, :]

                X_final_nondominated = np.vstack([Xnew, X_leftback_merged])
                Y_final_nondominated = np.vstack([Ynew, Y_leftback_merged])
            except ValueError:
                X_leftback_merged = None
                Y_leftback_merged = None

                X_final_nondominated = Xnew
                Y_final_nondominated = Ynew

        else:
            X_leftback_merged = None
            Y_leftback_merged = None

            X_final_nondominated = Xnew
            Y_final_nondominated = Ynew

        ii_final_nondominated = moo_problem._which_nondominated(
            X=X_final_nondominated,
            Y=Y_final_nondominated,
            dominated_tol=dominated_tol,
            verbose=verbose,
            max_checks=None,
            random_seed=None
        )
        X_final_nondominated = X_final_nondominated[ii_final_nondominated, :]
        Y_final_nondominated = Y_final_nondominated[ii_final_nondominated, :]

        history = {
            'X_list': X_list,
            'Y_list': Y_list,
            'allp_list': allp_list,
            'allG_list': allG_list,
            'allstop_list': allstop_list,
            'allstep_list': allstep_list,
            'allG_p_list': allG_p_list,
            'lastp_list': p_list,
            'lastG_list': G_list,
            'laststop_list': stop_list,
            'lastG_p_list': G_times_p_list,
            'X_leftback_merged': X_leftback_merged,
            'Y_leftback_merged': Y_leftback_merged,
            'X_final_nondominated': X_final_nondominated,
            'Y_final_nondominated': Y_final_nondominated
        }

        return Xnew, Ynew, history

