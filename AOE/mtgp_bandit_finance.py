
from email.policy import strict
import os
from this import d
import scipy, pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
import torch, random
import numpy as np
import matplotlib.pyplot as plt
from botorch.utils.transforms import standardize, normalize, unnormalize

from .plots import rescale_plot
from scipy.linalg import eigh

from .gp_bandit import gp_bandit, ExactGPModel, mtgp_bandit, MultitaskGPModel
from .gp_utils import Wasserstein_GP_mean

from .inducing_points_version.core.adaptive_regionalization import AdaptiveRegionalization_bandit

class mtgp_bandit_finance:
    def __init__(self, strategies, bandit_algo='TS', 
                            likelihood = gpytorch.likelihoods.GaussianLikelihood(),
                            train_x    = None,
                            train_i    = torch.zeros((0, 1), dtype=torch.long),
                            train_y    = torch.zeros(0, dtype=torch.float64),
                            bandit_params={'size_buffer': 50,
                                           'lambda': 1.,
                                           'size_window': 100,
                                           'threshold': 10,
                                           'delta': 0.6,
                                           },
                            training_iter = 10,
                            verbose       = False,
                            reinit        = True):
        self.deactivate_LR = False
        self.strategies    = strategies
        self.num_task      = len(self.strategies.keys())
        self.task_ids      = list(self.strategies.keys())
        self.training_iter = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo   = bandit_algo
        self.bandit_params = bandit_params
        self.verbose       =  verbose
        self.likelihood    = likelihood
        self.records       = {} # Traces of what happens
        self.reinit        = reinit

        if 'deactivate_LR' in bandit_params:
            self.deactivate_LR = True
        # optimisation of computations
        self.last_change_point_test_results = None
        #for algo_name in ('TS_NS', 'UCB_NS', 'TS_WAS', 'UCB_WAS', 'MAB_UCB', 'UCB_LR', 'RANDOM'):
        #    self.last_change_point_test_results[algo_name] = {}

        # create the mtgp
        if train_x is None:        train_x   = torch.zeros((0, self.num_task), dtype=torch.float64)

        self.mtgp = mtgp_bandit(likelihood,
                                bandit_algo,
                                self.num_task,
                                self.num_task, # we suppose 1D features for every task
                                train_x,
                                train_i, # indicator for tasks
                                train_y,
                                bandit_params = bandit_params,
                                training_iter = training_iter)

        
        if bandit_algo == 'UCB_LR':
            self.size_window   = self.bandit_params['size_window']
            self.delta_I       = self.bandit_params['delta_I'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.check_type_II = self.bandit_params['check_type_II']

            if 'force_threshold' in self.bandit_params:
                self.force_threshold = self.bandit_params['force_threshold']
            else:
                self.force_threshold = None

            self.records['lr_statistic'] = [] #{s: [] for s in self.strategies.keys() }
            self.records['noise']        = [] #{s: [] for s in self.strategies.keys() } # this is noise in context but also noise that is task specific
            self.records['lengthscale']  = [] #{s: [] for s in self.strategies.keys() } # lengthscale of the RBF and that of the task kernel
            self.records['task_covar']   = [] # covar matrix for the task {s: [] for s in self.strategies.keys() }

    def select_best_strategy(self, features):
        # before testing the change points show the learnt GPs
        if self.verbose: 
            print('Plotting the GPs of strategies for the bandit: ', self.bandit_algo)
            self.plot_strategies()
            plt.show()

        if self.bandit_algo == 'UCB_LR':
            best_strat, best_ucb = "", -np.inf
            ucb_strats = self.compute_ucbs_lr(features)
            for (i_task, ucb_value) in enumerate(ucb_strats):
                if ucb_value > best_ucb:
                    best_strat, best_ucb = list(self.strategies.keys())[i_task], ucb_value

        else:
            best_strat = 'NOT IMPLEMENTED'
        return best_strat

    
    def update_data(self, features, strat, reward, retrain_hyperparameters = False):
        "One get reward, update data and retrain gp of the corresponding strat"
        #select feature
        feature_names = [self.strategies[_strat]['contextual_params']['feature_name'] for _strat in self.strategies.keys()]
        x_new = features[feature_names].values
        y_new = reward
        i_task = self.task_ids.index(strat)

        # the bandit is now eligible for change point detection
        #if strat in self.last_change_point_test_results[self.bandit_algo].keys():
        #    if self.verbose: print('I am deleting the change point results of ', strat)
        #    self.last_change_point_test_results[self.bandit_algo].pop(strat, None)
        if self.last_change_point_test_results is not None:
            if self.verbose: print('I am deleting the change point result')
            self.last_change_point_test_results = None

        #Add point to model
        if self.bandit_algo == 'UCB_LR':
            if self.verbose: print('adding : ', x_new, i_task, y_new, 'strat = ', strat)
            self.mtgp.update_data(x_new, i_task, y_new)

            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'RANDOM':
            # no need for points
            best_strat = 'OK'
        else:
            best_strat = 'NOT IMPLEMENTED'
        
        if retrain_hyperparameters:
            if self.bandit_algo not in ("RANDOM", "MAB_UCB", "MAB_TS"): #add exceptions here
                self.mtgp.train()

    def compute_ucb(self, features):
        "Compute ucb for one strat"
        if self.verbose: print('Computing UCB for the mtgp')
        t_x = torch.tensor([features]).double()
        return self.mtgp.compute_ucb(t_x)

    def compute_ucbs_lr(self, features):
        "Compute UCB for one strat with "
        if self.verbose: print('Computing UCB_LR for strategy:')   

        # record the current lenght scale and noise values
        self.records['noise']       += [np.concatenate((self.mtgp.model.likelihood.noise.detach().numpy(), 
                                                        self.mtgp.model.task_covar_module.raw_var.detach().numpy())) ]
        self.records['lengthscale'] += [self.mtgp.model.covar_module.lengthscale.detach().numpy()[0][0]]
        
        covar_matrix = self.mtgp.model.task_covar_module.covar_factor.detach().numpy() @ self.mtgp.model.task_covar_module.covar_factor.detach().numpy().T
        self.records['task_covar']   += [covar_matrix] # covar matrix for the task {s: [] for s in self.strategies.keys() }

        if self.verbose:
            print('Task specific correl matrix:')
            __diag = np.sqrt(np.diag(np.diag(covar_matrix)))
            __gaid = np.linalg.inv(__diag)
            __corl = __gaid @ covar_matrix @ __gaid
            print(__corl)

        # First Compute the LR stat
        
        if self.mtgp.model.train_targets.shape[0] > self.size_window:
            train_x, train_i = self.mtgp.model.train_inputs
            train_y = self.mtgp.model.train_targets
            
            #if self.bandit_algo in self.last_change_point_test_results.keys():
            if self.last_change_point_test_results is not None: 
                # it means no point has been added or reint has been done, so no need to redo test
                b_changepoint = self.last_change_point_test_results #[self.bandit_algo]
            else:
                try:
                    b_changepoint, lr_statistic, tau_I, delta_II = self.change_point_lr() #, tau_I 

                    if self.verbose: print('I am storing the change point test results for UCB_LR / MTGP')
                    #self.last_change_point_test_results[self.bandit_algo] =  b_changepoint
                    self.last_change_point_test_results =  b_changepoint
                    self.records['lr_statistic'] += [ (lr_statistic, tau_I, delta_II) ]
                except Exception as e:
                    print('Error while compute LR statistic:', str(e))
                    b_changepoint, lr_statistic, tau_I, delta_II = False, None,  None,  None
                

            if self.verbose:
                print('Plotting the GPs from both sub-windows')
                posterior_mean_1_1, posterior_covar_1_1, posterior_mean_1_2, posterior_covar_1_2, lower1_1, upper1_1, lower1_2, upper1_2,\
                    posterior_mean_2_1, posterior_covar_2_1, posterior_mean_2_2, posterior_covar_2_2, lower2_1, upper2_1, lower2_2, upper2_2\
                                = self.posterior_sliding_window_covar(n_test=30)

                if posterior_mean_1_1 != None:
                    self.plot_change_point(posterior_mean_1_1, posterior_covar_1_1, posterior_mean_1_2, posterior_covar_1_2, lower1_1, upper1_1, lower1_2, upper1_2,
                                            posterior_mean_2_1, posterior_covar_2_1, posterior_mean_2_2, posterior_covar_2_2, lower2_1, upper2_1, lower2_2, upper2_2, n_test=30)

            if b_changepoint:
                if not self.deactivate_LR:
                    if self.reinit:
                        self.mtgp = mtgp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        self.num_task,
                                                        self.num_task,
                                                        train_x       = train_x[-2:],
                                                        train_i       = train_i[-2:],
                                                        train_y       = train_y[-2:],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

                    else:
                        self.mtgp = mtgp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        self.num_task,
                                                        self.num_task,
                                                        train_x       = train_x[(- self.size_window//2):],
                                                        train_i       = train_i[(- self.size_window//2):],
                                                        train_y       = train_y[(- self.size_window//2):],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

                        self.mtgp.train() # train only if there are point
                
                # remove last test
                self.last_change_point_test_results = None
                #self.last_change_point_test_results[self.bandit_algo] = {}
                #self.last_change_point_test_results.pop(self.bandit_algo, None)

        # this should change and be updated to the case where context parameters are not always the same
        strat = self.task_ids[0]
        feature_names = [self.strategies[_strat]['contextual_params']['feature_name'] for _strat in self.strategies.keys()]
        t_x = torch.tensor([list(features[feature_names].values)]).double()

        if self.verbose: print('context value is:', features[self.strategies[strat]['contextual_params']['feature_name']])
        return self.mtgp.compute_ucbs(t_x)



    def plot_strategies(self, lv=None, uv=None, nb_strategies=2, n_test=100, xlabel=None):
        """Plot the fit of all strategies for sanity check"""
        
        #rescale_plot(W=W)
        f, axs = plt.subplots(1, nb_strategies, figsize=(2*nb_strategies,2), sharey=True)
        axs = np.array([axs])

        for ( (i_task,task), ax) in zip(enumerate(self.task_ids), np.ndarray.flatten(axs)):
            ax = self.mtgp.build_plot(ax, i_task = i_task, lv=lv, uv=uv, n_test = n_test, xlabel=xlabel)
            ax.set_title(task)
            # Get model and predictions
            #ax.tick_params(axis='x', rotation=90)

        return f, axs

    def change_point_lr(self):
        
        # Compute corresponding type 2 error
        # Compute Likelihood ratio test
        # If ratio test big enough reinitialize
        gp = self.mtgp
        train_x, train_i = gp.model.train_inputs
        train_y = gp.model.train_targets
        if gp.model.train_inputs[0].shape[0] < self.size_window:
            return False

        #train_x, test_x = gp.model.train_inputs[0][:self.p], gp.model.train_inputs[0][self.p:self.P]
        S_1, S_2 = train_x[(-self.size_window):(- self.size_window//2)], train_x[(-self.size_window//2):]
        y_1, y_2 = train_y[(-self.size_window):(- self.size_window//2)], train_y[(-self.size_window//2):]
        i_1, i_2 = train_i[(-self.size_window):(- self.size_window//2)], train_i[(-self.size_window//2):]
        
        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(S_1, i_1, y_1)
        likelihood_gp, model = gp.model.likelihood, gp.model
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_tilde = model(S_2, i_2)
            observed_pred = likelihood_gp(latent_f_tilde)
        mu_tilde, K_tilde = observed_pred.mean, observed_pred.covariance_matrix
        v_h0 = K_tilde + (likelihood_gp.noise**2)*torch.eye(self.size_window//2)

        # Compute K** mu** from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(train_x, train_i, train_y)
        
        likelihood_alt = gpytorch.likelihoods.GaussianLikelihood(noise  = likelihood_gp.noise) #noise=0.01
        #likelihood_alt              = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = torch.tensor([0.01]), 
        #                                                                                learn_additional_noise=False) 

        model =  MultitaskGPModel( ( torch.zeros((2, self.num_task), dtype=torch.float64), 
                                     torch.cat([torch.full((1,1), dtype=torch.long, fill_value=0), 
                                                torch.full((1,1), dtype=torch.long, fill_value=1)]),
                                     ), torch.zeros(2).double(), self.num_task, likelihood_alt)
        
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_star = model(S_2, i_2)
            observed_pred = likelihood_alt(latent_f_star)
        
        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        mu, K = observed_pred.mean, observed_pred.covariance_matrix
        v_h1  = K + (likelihood_alt.noise**2)*torch.eye(self.size_window//2)

        # Compute delta Vh0 Vh1
        v_h0_inv   = torch.inverse(v_h0)
        v_h1_inv   = torch.inverse(v_h1)
        delta      = v_h1_inv - v_h0_inv
        target_bis = y_2 - mu_tilde

        R = -y_2 @ v_h1_inv @ y_2 + target_bis @ v_h0_inv @ target_bis # the real LR stat is R - torch.log(torch.det(v_h1)) + torch.log(torch.det(v_h0))
        
        # Conpute Threshold based on type 1 error
        aux   = v_h0 @ delta
        mu_h0 = v_h0.shape[0] - mu_tilde @ v_h1_inv @ mu_tilde - torch.trace( v_h1_inv @ v_h0)

        sum_lambda_0  = torch.trace(aux @ aux)
        largest_eih0  = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_window//2 - 1, self.size_window//2 - 1))
        smallest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
        largest_eih0  = max(np.absolute(largest_eih0), np.absolute(smallest_eih0))

        tau_I = mu_h0 + max(torch.sqrt(-8*np.log(self.delta_I)*(sum_lambda_0 + mu_tilde @ v_h1_inv @ v_h0 @ v_h1_inv @ mu_tilde)), torch.tensor(-8*np.log(self.delta_I)*largest_eih0))

        # compute error II prob
        aux           = v_h1 @ delta

        #mu_h1 = -torch.trace(aux)
        mu_h1         = -v_h1.shape[0] + mu_tilde @ v_h0_inv @ mu_tilde + torch.trace( v_h0_inv @ v_h1)
        sum_lambda_1  = torch.trace(aux @ aux)
        largest_eih1  = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_window//2 - 1, self.size_window//2 - 1))
        smallest_eih1 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
        largest_eih1  = max(np.absolute(largest_eih1), np.absolute(smallest_eih1))

        # If delta_I is specified, abide by it, otherwise try to compute one
        if self.force_threshold is not None:
            error_II      = torch.max(torch.exp(-((mu_h1 - self.force_threshold)**2)/(8*(sum_lambda_1 + mu_tilde @ v_h0_inv @ v_h1 @ v_h0_inv @ mu_tilde))), torch.exp(-((mu_h1 - self.force_threshold)/(8*torch.tensor(largest_eih1))))) # = lambda 2            

            if self.verbose: print('I m forcing the threshold to ', self.force_threshold, ' and the LR stat is ', float(R), ' and tau_I is', float(tau_I))
            if self.check_type_II:
                return R >= self.force_threshold, R, tau_I, error_II
            else:
                return R >= self.force_threshold, R, tau_I, error_II
        
        else:
            error_II      = torch.max(torch.exp(-((mu_h1 - tau_I)**2)/(8*(sum_lambda_1 + mu_tilde @ v_h0_inv @ v_h1 @ v_h0_inv @ mu_tilde))), torch.exp(-((mu_h1 - tau_I)/(8*torch.tensor(largest_eih1))))) # = lambda 2                    
        
            if self.check_type_II:
                #In case significant test compute equivalent second threshold

                # If test inferior, we already know there is no change
                if R <= tau_I:
                    return False, R, tau_I, error_II
                #elif error_II < self.threshold:
                    #return True, R, tau_I, error_II
                else:
                    return True, R, tau_I, error_II
            else:
                #return R >= self.delta , R #, tau_I
                return R >= tau_I, R, tau_I, error_II
            


    def posterior_sliding_window_covar(self, n_test = 30, lv = -1, uv = 1, force_size_window = None):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        """
        gp = self.mtgp

        train_x, train_i = gp.model.train_inputs
        train_y = gp.model.train_targets
        
        #bounds = torch.tensor([[-1.0], [1.0]])
        #train_y_normalized =  normalize(train_y.detach(), bounds=bounds)
        if force_size_window:
            old_size_window   = self.size_window
            self.size_window  = force_size_window

        try:
            if train_y.shape[0] < self.size_window:
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        except Exception as e:
            self.size_window = train_y.shape[0]
            if train_y.shape[0] < self.size_window:
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        train_x_2 = train_x[(-self.size_window//2):]
        train_y_2 = train_y[(-self.size_window//2):]
        train_i_2 = train_i[(-self.size_window//2):]

        train_x_1 = train_x[(-self.size_window):(- self.size_window//2)]
        train_i_1 = train_i[(-self.size_window):(- self.size_window//2)]
        train_y_1 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        #o_gaussLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        #o_gaussLikelihood.noise = torch.tensor(0.0001)
        
        gp_1 = mtgp_bandit(self.likelihood,
                    self.bandit_algo,
                    self.num_task,
                    self.num_task,
                    train_x       = train_x_1,
                    train_i       = train_i_1,
                    train_y       = train_y_1,
                    bandit_params = self.bandit_params,
                    training_iter = self.training_iter)
        gp_1.train()

        model, likelihood = gp_1.model, gp_1.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # for each task (there are 2 here, code should be generalised)
            test_x       =  torch.cat( [torch.linspace(lv, uv, n_test) for i in range(gp_1.nb_features)]).reshape(gp_1.nb_features, n_test).T

            # task 1 :
            test_i_task   = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)
            observed_pred = likelihood(model( test_x, test_i_task ))
            posterior_mean_1_1  = observed_pred.mean
            posterior_covar_1_1 = observed_pred.covariance_matrix
            lower1_1, upper1_1    = observed_pred.confidence_region()

            # task 2 :
            test_i_task   = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=1)
            observed_pred = likelihood(model( test_x, test_i_task ))
            posterior_mean_1_2    = observed_pred.mean
            posterior_covar_1_2   = observed_pred.covariance_matrix
            lower1_2, upper1_2    = observed_pred.confidence_region()

        gp_2 = mtgp_bandit(self.likelihood,
                    self.bandit_algo,
                    self.num_task,
                    self.num_task,
                    train_x       = train_x_2,
                    train_i       = train_i_2,
                    train_y       = train_y_2,
                    bandit_params = self.bandit_params,
                    training_iter = self.training_iter)
        gp_2.train()
        
        model, likelihood = gp_2.model, gp_2.model.likelihood
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # for each task (there are 2 here, code should be generalised)
            test_x       =  torch.cat( [torch.linspace(lv, uv, n_test) for i in range(gp_2.nb_features)]).reshape(gp_2.nb_features, n_test).T
            
            # task 1 :
            test_i_task   = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)
            observed_pred = likelihood(model( test_x, test_i_task ))
            posterior_mean_2_1  = observed_pred.mean
            posterior_covar_2_1 = observed_pred.covariance_matrix
            lower2_1, upper2_1    = observed_pred.confidence_region()

            # task 2 :
            test_i_task   = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=1)
            observed_pred = likelihood(model( test_x, test_i_task ))
            posterior_mean_2_2    = observed_pred.mean
            posterior_covar_2_2   = observed_pred.covariance_matrix
            lower2_2, upper2_2    = observed_pred.confidence_region()

        if force_size_window:
            self.size_window = old_size_window

        return posterior_mean_1_1, posterior_covar_1_1, posterior_mean_1_2, posterior_covar_1_2, lower1_1, upper1_1, lower1_2, upper1_2,\
                posterior_mean_2_1, posterior_covar_2_1, posterior_mean_2_2, posterior_covar_2_2, lower2_1, upper2_1, lower2_2, upper2_2
 


    def plot_change_point(self, posterior_mean_1_1, posterior_covar_1_1, posterior_mean_1_2, posterior_covar_1_2, lower1_1, upper1_1, lower1_2, upper1_2,\
                                 posterior_mean_2_1, posterior_covar_2_1, posterior_mean_2_2, posterior_covar_2_2, lower2_1, upper2_1, lower2_2, upper2_2,
                                n_test=30, ax1=None, ax2=None, force_size_window = None):
        if force_size_window:
            old_size_window   = self.size_window
            self.size_window  = force_size_window

        #print(f'** performing change point test for strategy : {strat} **')
        #print('Distance: ', round(d, 5), ' Threshold:', self.b )
        gp      = self.mtgp
        train_x, train_i = gp.model.train_inputs
        train_y = gp.model.train_targets

        #train_x, test_x = gp.model.train_inputs[0][:self.p], gp.model.train_inputs[0][self.p:self.P]
        xs_1, xs_2 = train_x[(-self.size_window):(- self.size_window//2)], train_x[(-self.size_window//2):]
        ys_1, ys_2 = train_y[(-self.size_window):(- self.size_window//2)], train_y[(-self.size_window//2):]
        is_1, is_2 = train_i[(-self.size_window):(- self.size_window//2)], train_i[(-self.size_window//2):]
        
        if self.verbose:
            print('Average performance GP1:', ys_1.mean())
            print('Average performance GP2:', ys_2.mean())

        test_x = torch.linspace(train_x.min().item(), train_x.max().item(), n_test).double()
        if ax1 is None: fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        # first task
        ax1.plot(test_x.numpy(), posterior_mean_1_1.numpy(), 'lightcoral')
        ax1.plot(test_x.numpy(), posterior_mean_2_1.numpy(), 'lightblue')
        ax1.plot(xs_1[is_1.flatten()==0, 0].numpy(), ys_1[is_1.flatten()==0].numpy(), 'r*')
        ax1.plot(xs_2[is_2.flatten()==0, 0].numpy(), ys_2[is_2.flatten()==0].numpy(), 'b*')

        # second task
        i_task = 1
        ax2.plot(test_x.numpy(), posterior_mean_1_2.numpy(), 'lightcoral', label=r"$\overline \mathcal W$")
        ax2.plot(test_x.numpy(), posterior_mean_2_2.numpy(), 'lightblue',  label=r"$\overline \mathcal W$")
        ax2.plot(xs_1[is_1.flatten()==i_task, 1].numpy(), ys_1[is_1.flatten()==i_task].numpy(), 'r*')
        ax2.plot(xs_2[is_2.flatten()==i_task, 1].numpy(), ys_2[is_2.flatten()==i_task].numpy(), 'b*')

        for ax in (ax1, ax2):
            #ax.yaxis.tick_right()
            #ax.yaxis.set_label_position("left") 
            #ax.grid(axis='both')
            ax.set_axisbelow(True)         #,  linestyle='-', linewidth=0.5
            # ax.set_ylabel('Reward in ticks')
            ax.legend([r"GP1", r"GP2"], handlelength=0.2, framealpha=0.2, loc='best', ncol=2)
        
        ax1.set_xlabel(list(self.strategies.keys())[0])
        ax2.set_xlabel(list(self.strategies.keys())[1])

        ax1.fill_between(test_x.numpy(), lower1_1.detach().numpy(), upper1_1.detach().numpy(), alpha=0.4, color='lightcoral')
        ax1.fill_between(test_x.numpy(), lower2_1.detach().numpy(), upper2_1.detach().numpy(), alpha=0.4, color='lightblue')

        ax2.fill_between(test_x.numpy(), lower1_2.detach().numpy(), upper1_2.detach().numpy(), alpha=0.4, color='lightcoral')
        ax2.fill_between(test_x.numpy(), lower2_2.detach().numpy(), upper2_2.detach().numpy(), alpha=0.4, color='lightblue')
        if ax1 is None: 
            plt.tight_layout()
            plt.show()

        if force_size_window:
            self.size_window = old_size_window
