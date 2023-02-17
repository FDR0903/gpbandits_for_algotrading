"""
Edoardo Caldarelli, ETH Zurich
"""

import numpy as np
import gpflow
import tensorflow as tf
from ..utils.stat_test import StatisticalTest
np.set_printoptions(threshold=np.inf)
#from gpflow import features
from gpflow import covariances as features
from gpflow.logdensities import multivariate_normal
import tensorflow_probability as tfp

gpflow.config.set_default_float(tf.float32)

def updateLengthScaleSigmoid(o_, lb, ub):
    old_parameter = o_.lengthscales
    new_parameter = gpflow.Parameter(
        old_parameter,
        trainable = old_parameter.trainable,
        prior     = old_parameter.prior,
        name      = old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
        transform = tfp.bijectors.Sigmoid(tf.cast(lb, gpflow.default_float()), 
                                          tf.cast(ub, gpflow.default_float()), 
                                          validate_args=True),
    )
    o_.lengthscales = new_parameter

def updateVarianceSigmoid(o_, lb, ub):
    old_parameter = o_.lengthscales
    new_parameter = gpflow.Parameter(
        old_parameter,
        trainable = old_parameter.trainable,
        prior     = old_parameter.prior,
        name      = old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
        transform = tfp.bijectors.Sigmoid(tf.cast(lb, gpflow.default_float()), 
                                          tf.cast(ub, gpflow.default_float()), 
                                          validate_args=True),
    )
    o_.variance = new_parameter

class AdaptiveRegionalization(object):
    """
    Class that regionalizes the time domain in a streaming fashion.
    """

    def __init__(self, domain_data,
                 system_data,
                 delta,
                 min_w_size,
                 n_ind_pts,
                 #seed,
                 kern="RBF",
                 domain_test=None,
                 system_test=None,
                 input_horizon=None):
        """
        Constructor
        :param domain_data: [n x 1] array of timesteps;
        :param system_data: [n x 1] array of observations;
        :param delta: the delta hyperparameter to be used in the thresholds;
        :param min_w_size: the minimum window size allowed;
        :param n_ind_pts: the number of inducing points to use;
        :param seed: the seed (fixed for reproducibility);
        :param n_batches: the number of batches in which the overll trajectory is partitioned;
        :param kern: the kernel to be used.
        """
        self.x = domain_data
        self.y = system_data
        self.delta = delta
        self.min_window_size = min_w_size
        self.num_inducing_points = n_ind_pts
        #self.seed = seed
        self.kern = kern
        self.domain_test = domain_test
        self.y_test = system_test
        # This param is needed to decouple the minimum window size from a reduced time horizon
        # (w.r.t. the final value of the trajectory).
        self.input_horizon = input_horizon
        if self.input_horizon is not None:
            self._slice_domain_function()

    def _slice_domain_function(self):
        sliced_x_y = np.array([e for e in np.column_stack((self.x, self.y)) if e[0] <= self.input_horizon])
        self.x = np.expand_dims(sliced_x_y[:, 0], axis=-1)
        self.y = np.expand_dims(sliced_x_y[:, 1], axis=-1)


    def _create_expert(self, window, new: bool, x_mean=None, x_std=None, y_mean=None, y_std=None):
        """
        This method creates the expert on the region of interest (full window or overlap);
        :param window: the slice of data that supports the expert;
        :param new: whether the expert is trained on the overlap or not;
        :param x_mean: the mean to use in time standardization;
        :param x_std: the std dev to use in time standardization;
        :param y_mean: the mean to use in observations' standardization;
        :param y_std: the std dev to use in observations' standardization;
        :return: the expert, together with the (potentially recomputed) time and observations' mean, atd dev.
        """
        x = np.expand_dims(window[:, 0], axis=-1)
        y = np.expand_dims(window[:, 1], axis=-1)
        if not new:
            x_mean = np.mean(x)
            x_std  = np.std(x)
            y_mean = np.mean(y)
            y_std  = np.std(y)
        x -= x_mean
        x /= x_std
        y -= y_mean
        y /= y_std
        z_init = np.random.choice(x[:, 0], min(self.num_inducing_points, x.shape[0]), replace=False)
        z_init = np.expand_dims(z_init, axis=-1)

        #with gpflow.decors.defer_build():
        if self.kern == "RBF":
            k = gpflow.kernels.RBF()
            updateLengthScaleSigmoid(k, 1e-3, 100)
#            k.lengthscales.transform = tensorflow_probability.distributions.Logistic(1e-3, 100)
        elif self.kern == "Matern52":
            print(self.kern)
            k = gpflow.kernels.Matern52()
            updateLengthScaleSigmoid(k, 1e-3, 100)
#            k.lengthscales.transform = tensorflow_probability.distributions.Logistic(1e-3, 100)
        elif self.kern == "RQ":
            print(self.kern)
            k = gpflow.kernels.RationalQuadratic()
            updateLengthScaleSigmoid(k, 1e-3, 100)
#            k.lengthscales.transform = tensorflow_probability.distributions.Logistic(1e-3, 100)
        elif self.kern == "Periodic":
            base = gpflow.kernels.RBF()
            if not new:
#                base.variance.transform = tensorflow_probability.distributions.Logistic(1e-8, 1.4)
                updateVarianceSigmoid(base, 1e-8, 1.4)
            updateLengthScaleSigmoid(base, 1e-3, 100)            
#            base.lengthscales.transform = tensorflow_probability.distributions.Logistic(1e-3, 100)

            k = gpflow.kernels.Periodic(base=base)
        elif self.kern == "Linear":
            k = gpflow.kernels.Linear()

        xx = tf.cast(x, gpflow.default_float())
        yy = tf.cast(y, gpflow.default_float())
        zz_init = tf.cast(z_init, gpflow.default_float())
        expert = gpflow.models.SGPR( data=(xx, yy), kernel=k, inducing_variable=zz_init) #Z=z_init)
        # end of with
        # noise_variance

        k.variance = 1.0
        if not new and self.kern != "Periodic":
            updateVarianceSigmoid(k, 1e-8, 1.4)
            # k.variance.transform = tensorflow_probability.distributions.Logistic(1e-8, 1.4)
            # gpflow.transforms.Logistic(1e-8, 1.4)
        
        # expert.compile()
        xx_mean = tf.cast(x_mean, gpflow.default_float())
        xx_std  = tf.cast(x_std, gpflow.default_float())
        yy_mean = tf.cast(y_mean, gpflow.default_float())
        yy_std  = tf.cast(y_std, gpflow.default_float())
        return expert, xx_mean, xx_std, yy_mean, yy_std

    def _build_likelihood(self, model: gpflow.models.SGPR):
        """
        This method builds the likelihood of the given model.
        """
        # gpflow.models.GPR
        K_uf = gpflow.get_default_session().run(features.Kuf(model.feature, model.kern, model.data[0].value))
        K_uu = gpflow.get_default_session().run(
            features.Kuu(model.feature, model.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv = np.linalg.inv(K_uu)
        K = np.matmul(np.matmul(np.transpose(K_uf), K_uu_inv), K_uf) + np.identity(model.data[0].value.shape[0],
                                                                                   dtype=gpflow.default_float()) \
            * model.likelihood.variance.value
        L = np.linalg.cholesky(K)
        m = model.mean_function(model.data[0].value)
        y_tensor = tf.constant(model.data[1].value)
        logpdf   = gpflow.get_default_session().run(
            multivariate_normal(y_tensor, m, L))  # (R,) log-likelihoods for each independent dimension of Y

        return np.sum(logpdf)

    def _build_norm_const(self, model_1: gpflow.models.SGPR, model_2: gpflow.models.SGPR):
        """
        This method builds the normalization constant of the product of two Gaussian pdfs.
        """
        # gpflow.models.GPR
        K_uf_1 = gpflow.get_default_session().run(features.Kuf(model_1.feature, model_1.kern, model_1.data[0].value))
        K_uu_1 = gpflow.get_default_session().run(
            features.Kuu(model_1.feature, model_1.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv_1 = np.linalg.inv(K_uu_1)
        K_1 = np.matmul(np.matmul(np.transpose(K_uf_1), K_uu_inv_1), K_uf_1) + np.identity(model_1.data[0].value.shape[0],
                                                                                           dtype=gpflow.default_float()) \
              * model_1.likelihood.variance.value

        K_uf_2 = gpflow.get_default_session().run(features.Kuf(model_2.feature, model_2.kern, model_2.data[0].value))
        K_uu_2 = gpflow.get_default_session().run(
            features.Kuu(model_2.feature, model_2.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv_2 = np.linalg.inv(K_uu_2)
        K_2 = np.matmul(np.matmul(np.transpose(K_uf_2), K_uu_inv_2), K_uf_2) + np.identity(model_2.data[0].value.shape[0],
                                                                                           dtype=gpflow.default_float()) \
              * model_2.likelihood.variance.value

        L = np.linalg.cholesky(K_1 + K_2)
        m_1 = model_1.mean_function(model_1.data[0].value)
        m_2 = model_2.mean_function(model_2.data[0].value)
        logpdf = gpflow.get_default_session().run(
            multivariate_normal(m_1, m_2, L))  # (R,) log-likelihoods for each independent dimension of Y

        return np.sum(logpdf)

    def _build_norm_const_new(self, model_1: gpflow.models.SGPR):
        """
        This method builds the normalization constant of a given model.
        """
        # gpflow.models.GPR
        K_uf_1 = gpflow.get_default_session().run(features.Kuf(model_1.feature, model_1.kern, model_1.data[0].value))
        K_uu_1 = gpflow.get_default_session().run(
            features.Kuu(model_1.feature, model_1.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv_1 = np.linalg.inv(K_uu_1)
        K_1 = np.matmul(np.matmul(np.transpose(K_uf_1), K_uu_inv_1), K_uf_1) + np.identity(model_1.data[0].value.shape[0],
                                                                                           dtype=gpflow.default_float()) \
              * model_1.likelihood.variance.value
        _, c = np.linalg.slogdet(a=2 * np.pi * K_1)
        return 0.5 * c

    def test(self):
        final_pred = np.empty((0, 1))
        final_gt = np.empty((0, 1))
        final_time = np.empty((0, 1))
        for region in self.closed_windows:
            window_test = np.array(
                [e for e in np.column_stack((self.domain_test, self.y_test)) if
                 region["window_start"] <= e[0] < region["window_end"]])

            model_test, x_mean_test, x_std_test, y_mean_test, y_std_test = self._create_expert(window_test, False)
            #opt_test = gpflow.train.ScipyOptimizer()
            #opt_test.minimize(model_test)

            opt_test = gpflow.optimizers.Scipy()
            opt_test.minimize(model_test.training_loss, 
                                       variables=model_test.trainable_variables)

            x_pred_test = np.expand_dims(window_test[:, 0], axis=-1)
            pred, _ = model_test.predict_f(x_pred_test)
            pred = pred * y_std_test + y_mean_test
            y_gt = np.expand_dims(window_test[:, 1], axis=-1) * y_std_test + y_mean_test

            final_pred = np.concatenate((final_pred, pred), axis=0)
            final_gt = np.concatenate((final_gt, y_gt), axis=0)
            final_time = np.concatenate((final_time, x_pred_test * x_std_test + x_mean_test), axis=0)
            #gpflow.reset_default_graph_and_session()

        #gpflow.reset_default_graph_and_session()

        self.rmse = [self.x.shape[0], np.sqrt(np.sum((final_pred - final_gt) ** 2) / final_gt.shape[0])]

    def regionalize(self) -> None:
        """
        This method applies ADAGA streaming GP regression.
        """
        start = self.x[0, 0]
        end = start + 2 * self.min_window_size # + self.batch_time_jump
        close_current_window = False
        new_window = True
        while True:
            #gpflow.reset_default_graph_and_session()
            #tf.set_random_seed(self.seed)

            window = np.array([e for e in np.column_stack((self.x, self.y)) if start <= e[0] < end])
            print("start, end:", start, end)
            print("WINDOW SHAPE", window.shape)

            if window.shape[0] <= 1:
                break

            best_start_new_exp = end - self.min_window_size

            window_current_expert = np.array([e for e in window if start <= e[0] < end])
            model_current_expert, x_mean, x_std, y_mean, y_std = self._create_expert(window_current_expert, False)
#            opt_current_expert = gpflow.train.ScipyOptimizer()
#            opt_current_expert.minimize(model_current_expert)
            opt_current_expert = gpflow.optimizers.Scipy()
            opt_current_expert.minimize(model_current_expert.training_loss, 
                               variables=model_current_expert.trainable_variables)

            if min(end, self.x[-1, 0]) - start > self.min_window_size + 3 > end - self.x[-1, 0]:

                window_new_expert = np.array([e for e in window if best_start_new_exp <= e[0] < end])
                model_new_expert, _, _, _, _ = self._create_expert(window_new_expert, True, x_mean, x_std, y_mean, y_std)
                model_current_expert.data[0] = model_new_expert.data[0].value
                model_current_expert.data[1] = model_new_expert.data[1].value
                #opt_new_expert = gpflow.train.ScipyOptimizer()
                #opt_new_expert.minimize(model_new_expert)
                opt_new_expert = gpflow.optimizers.Scipy()
                opt_new_expert.minimize(model_new_expert.training_loss, 
                                        variables = model_new_expert.trainable_variables)

                print("CURRENT MODEL", model_current_expert.as_pandas_table())

                print("NEW MODEL", model_new_expert.as_pandas_table())
                statistical_test = StatisticalTest(model_current_expert, model_new_expert, self.delta)
                bad_current_window = statistical_test.test(gpflow.get_default_session())

                if bad_current_window:
                    close_current_window = True

            if not close_current_window or new_window:
                new_window = False

            if end > self.x[-1] or close_current_window:
                new_window = True
                end_test = self.x[-1, 0] if end > self.x[-1] else end - self.min_window_size

                self.closed_windows.append(
                    {"window_start": start, "window_end": end_test})

                start = end - self.min_window_size

                if end > self.x[-1]:
                    break
                end = start + 2 * self.min_window_size


            if not close_current_window or end - start < self.min_window_size or new_window:
                end += self.batch_time_jump

            close_current_window = False
        print("PARTITIONING CREATED:", [(e["window_start"], e["window_end"]) for e in self.closed_windows])
        #gpflow.reset_default_graph_and_session()

class AdaptiveRegionalization_bandit(AdaptiveRegionalization):
    def __init__(self, domain_data,
                 system_data,
                 delta,
                 min_w_size,
                 n_ind_pts,
                 kern="RBF",
                 domain_test=None,
                 system_test=None,
                 input_horizon=None):
        super().__init__(domain_data,
                 system_data,
                 delta,
                 min_w_size,
                 n_ind_pts,
                 kern,
                 domain_test,
                 system_test,
                 input_horizon)


    def regionalize(self) -> None:
    
        #gpflow.reset_default_graph_and_session()
        #tf.set_random_seed(self.seed)


        ## Build expert on full window and train

        window_current_expert = np.array([e for e in np.column_stack((self.x, self.y))])
        model_current_expert, x_mean, x_std, y_mean, y_std = self._create_expert(window_current_expert, False)
        #opt_current_expert = gpflow.train.ScipyOptimizer()
        #opt_current_expert.minimize(model_current_expert)
        opt_current_expert = gpflow.optimizers.Scipy()
        opt_current_expert.minimize(model_current_expert.training_loss, 
                               variables=model_current_expert.trainable_variables)

        
        ## Build expert on closest window and train
        window_new_expert = np.array([e for e in np.column_stack((self.x[(-self.min_window_size//2):], self.y[(-self.min_window_size//2):]))])
        window_new_expert = tf.cast(window_new_expert, gpflow.default_float())
        model_new_expert, _, _, _, _ = self._create_expert(window_new_expert, True, x_mean, x_std, y_mean, y_std)
        model_current_expert.data = model_new_expert.data
        
        #opt_new_expert = gpflow.train.ScipyOptimizer()
        #opt_new_expert.minimize(model_new_expert)
        #model_new_expert.trainable_variables = (tf.cast(model_new_expert.trainable_variables[i], 
        #                                                gpflow.default_float()) for i in range(len(model_new_expert.trainable_variables)))

        opt_new_expert = gpflow.optimizers.Scipy()
        opt_new_expert.minimize(model_new_expert.training_loss, 
                                variables=model_new_expert.trainable_variables)

        ## Perform Statistical test
        statistical_test = StatisticalTest(model_current_expert, model_new_expert, self.delta)
        bad_current_window = statistical_test.test()
        #bad_current_window = statistical_test.test(tf.get_default_session())
        #gpflow.reset_default_graph_and_session()


        if bad_current_window:
            return True
        else:
            return False
        
            