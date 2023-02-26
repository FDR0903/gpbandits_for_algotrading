
from email.policy import strict
import os
from this import d
import scipy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
import torch, random
import numpy as np
import matplotlib.pyplot as plt
from .plots import rescale_plot
from scipy.linalg import eigh

from .gp_bandit import gp_bandit, ExactGPModel
from .gp_utils import Wasserstein_GP_mean

from .inducing_points_version.core.adaptive_regionalization import AdaptiveRegionalization_bandit

class gp_bandit_finance:
    def __init__(self, strategies, bandit_algo='TS', 
                            likelihood = gpytorch.likelihoods.GaussianLikelihood(),
                            train_x = torch.zeros((0, 1), dtype=torch.float64),
                            train_y = torch.zeros(0, dtype=torch.float64),
                            bandit_params={'size_buffer': 50,
                                           'lambda': 1.,
                                           'size_window': 100,
                                           'threshold': 10,
                                           'delta': 0.6,
                                           },
                            training_iter = 10,
                            verbose       = False,
                            reinit        = True):

        self.strategies    = strategies
        self.training_iter = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo   = bandit_algo
        self.bandit_params = bandit_params
        self.verbose       =  verbose
        self.likelihood    = likelihood
        self.reinit    = likelihood


        if bandit_algo == 'TS_NS':
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'UCB_NS':
            self.size_buffer   = self.bandit_params['size_buffer']
            self.lamb          = self.bandit_params['lambda']
        elif bandit_algo == 'TS_WAS':
            self.size_window   = self.bandit_params['size_window']
            self.b             = self.bandit_params['threshold'] ### Must be even to divide last points by two
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'UCB_WAS':
            self.size_window   = self.bandit_params['size_window']
            self.b             = self.bandit_params['threshold'] ### Must be even to divide last points by two
            self.lamb          = self.bandit_params['lambda']
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'TS_ADAGA':
            self.size_window   = self.bandit_params['size_window']
            self.delta         = self.bandit_params['delta'] ### Bound on type 1 error
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'UCB_ADAGA':
            self.size_window   = self.bandit_params['size_window']
            self.delta         = self.bandit_params['delta'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'MAB_UCB':
            self.size_window   = self.bandit_params['size_window']
            self.delta         = self.bandit_params['delta'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.strat_t       = {} ### Dictionary of time since reset for ucb calculation
            for strat in self.strategies.keys():
                self.strat_t[strat] = 0
        elif bandit_algo == 'MAB_TS':
            self.size_window   = self.bandit_params['size_window']
            self.delta         = self.bandit_params['delta'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.strat_t       = {} ### Dictionary of time since reset for ucb calculation
            for strat in self.strategies.keys():
                self.strat_t[strat] = 0
        elif bandit_algo == 'UCB_LR':
            self.size_window   = self.bandit_params['size_window']
            self.delta         = self.bandit_params['delta'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
        
        self.strat_gp_dict = {}
        for strat in self.strategies.keys():
            self.strat_gp_dict[strat] = gp_bandit(likelihood,
                                                    bandit_algo,
                                                    train_x,
                                                    train_y,
                                                    bandit_params = bandit_params,
                                                    training_iter = training_iter)

    def select_best_strategy(self, features):
        if self.bandit_algo == 'UCB_NS':
            "Compute ucb for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat

        elif self.bandit_algo == 'TS_NS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ts(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat

        elif self.bandit_algo == 'UCB':
            "Compute ucb for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat

        elif self.bandit_algo == 'TS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ucb(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat

        elif self.bandit_algo == 'TS_WAS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ts_was(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat

        elif self.bandit_algo == 'UCB_WAS':
            "Compute ts for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb_was(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat

        elif self.bandit_algo == 'TS_ADAGA':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ts_ada(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat
        elif self.bandit_algo == 'UCB_ADAGA':
            "Compute ucb for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb_ada(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat
        elif self.bandit_algo == 'RANDOM':
            "Compute ts for each gp and select best"
            best_strat, best_ts = random.choice(list(self.strategies.keys())), -np.inf
        elif self.bandit_algo == 'MAB_UCB':
            "Compute ucb for classical mab"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                self.strat_t[strat] += 1
                ucb_strat = self.compute_ucb_mab(strat)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat
        elif self.bandit_algo == 'MAB_TS':
            "Compute TS for classical mab"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                self.strat_t[strat] += 1
                ucb_strat = self.compute_ts(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat
        elif self.bandit_algo == 'UCB_LR':
            "Compute UCB"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb_lr(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat
        else:
            best_strat = 'NOT IMPLEMENTED'
        return best_strat

    def update_data(self, features, strat, reward, retrain_hyperparameters = False):
        "One get reward, update data and retrain gp of the corresponding strat"

        # Get model and data
        model = self.strat_gp_dict[strat]
        
        #select feature
        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
        y_new = reward

        #Add point to model
        if self.bandit_algo == 'UCB_NS':
            model.update_data_nst(x_new, y_new, self.size_buffer)

        elif self.bandit_algo == 'TS_NS':
            model.update_data_nst(x_new, y_new, self.size_buffer)

        elif self.bandit_algo == 'UCB':
            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'TS':
            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'TS_WAS':
            model.update_data_nst(x_new, y_new, self.size_buffer)
#            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'UCB_WAS':
            model.update_data_nst(x_new, y_new, self.size_buffer)
            # model.update_data(x_new, y_new)

        elif self.bandit_algo == 'TS_ADAGA':
            model.update_data(x_new, y_new)
#            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'UCB_ADAGA':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'MAB_UCB':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'MAB_TS':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'UCB_LR':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        else:
            best_strat = 'NOT IMPLEMENTED'
        

        if retrain_hyperparameters:
            model.train()

#    def update_data_nst(self, features, strat, reward, size_buffer, retrain_hyperparameters = False):
#        "One get reward, update data and retrain gp of the corresponding strat"#
#
        # Get model and data
#        model = self.strat_gp_dict[strat]
        
        #select feature
#        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
#        y_new = reward

        #Add point to model
#        model.update_data_nst(x_new, y_new, size_buffer)

#        if retrain_hyperparameters:
#            model.train()
    
    def compute_ucb(self, strat, features):
        "Compute ucb for one strat"
        if self.verbose: print('Computing UCB for strategy:', strat)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        return gp.compute_ucb(t_x)
    
    def compute_ts(self, strat, features):
        "Compute thompson sampling for one strat"
        if self.verbose: print('Computing thompson sampling for strategy:', strat)
        
        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        return gp.compute_ts(t_x)

    def compute_ts_was(self, strat, features):
        "Compute thompson sampling for one strat with was check first"
        if self.verbose: print('Computing thompson sampling waserstein for strategy:', strat)
        # First Compute the waser distancde
        #print('testing train target is', self.strat_gp_dict[strat].model.train_targets.shape[0], ' and window is ', self.size_window)
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets
            
            lv = train_x.min().item()
            uv = train_x.max().item()

            if self.change_point(strat, lv = lv, uv = uv):
                # update the gp
                if self.reinit:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)
                else:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(- self.size_window//2):],
                                                        train_y       = train_y[(- self.size_window//2):],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        
        return gp.compute_ts(t_x)

    def compute_ucb_was(self, strat, features):
        "Compute UCB for one strat with was check first"
        if self.verbose: print('Computing UCB Waserstein for strategy:', strat)   

        # First Compute the waser distance
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets
            
            lv = train_x.min().item()
            uv = train_x.max().item()
            
            if self.change_point(strat, lv = lv, uv = uv):
                if self.reinit:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)
                else:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(- self.size_window//2):],
                                                        train_y       = train_y[(- self.size_window//2):],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)
    

    def compute_ucb_ada(self, strat, features):
        "Compute thompson sampling for one strat with was check first"
        if self.verbose: print('Computing UCB ADA for strategy:', strat)   

        # First Compute the waser distance
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets

            if self.change_point_adaga(strat):
                if self.reinit:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)
                else:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(- self.size_window//2):],
                                                        train_y       = train_y[(- self.size_window//2):],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

                self.strat_gp_dict[strat].train()

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)
    
    def compute_ts_ada(self, strat, features):
        "Compute thompson sampling for one strat with was check first"
        if self.verbose: print('Computing thompson sampling ADA for strategy:', strat)   

        # First Compute the waser distancde
        #print('testing train target is', self.strat_gp_dict[strat].model.train_targets.shape[0], ' and window is ', self.size_window)
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets

            if self.change_point_adaga(strat):
                # update the gp
                if self.reinit:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)
                else:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(- self.size_window//2):],
                                                        train_y       = train_y[(- self.size_window//2):],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        
        return gp.compute_ts(t_x)

    def compute_ucb_mab(self, strat):
        "Compute ucb for one strat"
        if self.verbose: print('Computing UCB for strategy:', strat)


        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            
            gp = self.strat_gp_dict[strat]
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets

            train_x_1 = train_x[(-self.size_window//2):]
            train_y_1 = train_y[(-self.size_window//2):]
            train_y_2 = train_y[:(- self.size_window//2)]

            test_kol = scipy.stats.ks_2samp(train_y_2.detach().cpu().numpy(), train_y_1.detach().cpu().numpy(), alternative='less')
            print(f"p value test:{test_kol}")
            if test_kol[1] < 0.05: # Parameter to change p value threshold
                # update the gp
                if self.reinit:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)
                    self.strat_t[strat] = 0 ## Reinitialize bandit as if just existed

                else:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                            self.bandit_algo,
                                                            train_x       = train_x_1,
                                                            train_y       = train_y_1,
                                                            bandit_params = self.bandit_params,
                                                            training_iter = self.training_iter)
                    self.strat_t[strat] = self.size_window//2 ## Reinitialize bandit as if just existed
        
        gp = self.strat_gp_dict[strat]
        train_y = gp.model.train_targets
        if train_y.shape[0] == 0:
            return np.inf
        else:
            return np.mean(train_y.detach().cpu().numpy()) + np.sqrt(2*np.log(self.strat_t[strat])/train_y.shape[0])

    def compute_ucb_lr(self, strat, features):
        "Compute UCB for one strat with "
        if self.verbose: print('Computing UCB LR for strategy:', strat)   

        # First Compute the waser distance
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets

            if self.change_point_lr(strat):
                if self.reinit:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

                else:
                    self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(- self.size_window//2):],
                                                        train_y       = train_y[(- self.size_window//2):],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

                self.strat_gp_dict[strat].train()

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)



    def plot_strategies(self, strats = "all", lv=-1, uv=1, n_test=100, xlabel=None):
        """Plot the fit of all strategies for sanity check"""
        
        if strats == "all":
            nb_strategies = len(self.strategies.keys())
            strats = self.strategies.keys()
        else: nb_strategies = len(strats)

        #rescale_plot(W=W)
        f, axs = plt.subplots(1, nb_strategies, figsize=(4*nb_strategies,3), sharey=True)
        axs = np.array([axs])
        for ((_, strat), ax) in zip(enumerate(strats), np.ndarray.flatten(axs)):
            gp = self.strat_gp_dict[strat]
            ax = gp.build_plot(ax, lv=lv, uv=uv, n_test = n_test, xlabel=xlabel)
            ax.set_title(strat)
            # Get model and predictions
            #ax.tick_params(axis='x', rotation=90)

        return f, axs

    def posterior_sliding_window_confidence(self, strat, n_test = 100, lv = -1, uv = 1, training_iter=50):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        """
        error_on_purpose += 1
        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        try:
            if train_y.shape[0] < self.size_window:
                return None, None, None, None
        except Exception as e:
            self.size_window = train_y.shape[0]
            if train_y.shape[0] < self.size_window:
                return None, None, None, None

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        gp_1 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_1,
                        train_y_1,
                        bandit_params = self.bandit_params,
                        training_iter = training_iter)

        gp_1.train()

        model, likelihood = gp_1.model, gp_1.model.likelihood
        # Put the model into evaluation mode
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()

            # Make predictions by feeding model through likelihood
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        lower1, upper1 = observed_pred.confidence_region()

        ### Second posterior distribution
        gp_2 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_2,
                        train_y_2,
                        bandit_params = self.bandit_params,
                        training_iter = training_iter)

        gp_2.train()

        model, likelihood = gp_2.model, gp_2.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_2 = observed_pred.mean
        lower2, upper2 = observed_pred.confidence_region()

        return posterior_mean_1, lower1, upper1, posterior_mean_2, lower2, upper2

    def posterior_sliding_window_covar(self, strat, n_test = 30, lv = -1, uv = 1):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        """
        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        try:
            if train_y.shape[0] < self.size_window:
                return None, None, None, None
        except Exception as e:
            self.size_window = train_y.shape[0]
            if train_y.shape[0] < self.size_window:
                return None, None, None, None

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        o_gaussLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        o_gaussLikelihood.noise = 0.0001

        gp_1 = gp_bandit(o_gaussLikelihood,
                        self.bandit_algo,
                        train_x_1,
                        train_y_1,
                        bandit_params = self.bandit_params,
                        training_iter = self.training_iter)

        gp_1.train()

        model, likelihood = gp_1.model, gp_1.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
            posterior_mean_1  = observed_pred.mean
            posterior_covar_1 = observed_pred.covariance_matrix
            lower1, upper1    = observed_pred.confidence_region()

        ### Second posterior distribution
        o_gaussLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        o_gaussLikelihood.noise = 0.0001
        gp_2 = gp_bandit(o_gaussLikelihood,
                        self.bandit_algo,
                        train_x_2,
                        train_y_2,
                        bandit_params = self.bandit_params,
                        training_iter = self.training_iter)

        gp_2.train()

        model, likelihood = gp_2.model, gp_2.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred     = likelihood(model(test_x))
            posterior_mean_2  = observed_pred.mean
            posterior_covar_2 = observed_pred.covariance_matrix
            lower2, upper2    = observed_pred.confidence_region()

        return posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2, lower1, upper1, lower2, upper2

    def change_point(self, strat, n_test = 100, lv = -1, uv = 1):
        """Test for change point detection

        Args:
            strat (str): strategy to test

        Return:
            bool: whether chage point is detected
        """        
        posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2,\
                lower1, upper1, lower2, upper2 = self.posterior_sliding_window_covar(strat, n_test, lv, uv)

        if (posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2) == (None, None, None, None):
            return False

        ### Compute wasserstein distance between gp:
        #distance_mean = (posterior_mean_1 - posterior_mean_2).pow(2).sum()
        d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), posterior_mean_2.numpy(), posterior_covar_2.numpy())
        
#        posterior_mean_1, lower1, upper1,\
#        posterior_mean_2, lower2, upper2 = self.posterior_sliding_window_confidence(strat)

        # if True:
        #  HERE : plot both 
        if self.verbose:
            print(f'** performing change point test for strategy : {strat} **')
            print('Distance: ', round(d, 5), ' Threshold:', self.b )
    
            train_x = self.strat_gp_dict[strat].model.train_inputs[0]
            train_y = self.strat_gp_dict[strat].model.train_targets
            
            train_x_1 = train_x[(-self.size_window//2):]
            train_y_1 = train_y[(-self.size_window//2):]
            train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
            train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]

            print('Average performance GP1:', train_y_1.mean())
            print('Average performance GP2:', train_y_2.mean())

            test_x = torch.linspace(train_x.min().item(), train_x.max().item(), 100).double()
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            for ax in (ax1, ax2):
                ax.plot(test_x.numpy(), posterior_mean_1.numpy(), 'k', label="window_1")
                ax.plot(train_x_1.numpy(), train_y_1.numpy(), 'k*')

                ax.plot(test_x.numpy(), posterior_mean_2.numpy(), 'b', label="window_2")
                ax.plot(train_x_2.numpy(), train_y_2.numpy(), 'b*')
                #ax.set_title(f"Strategy: " + strat +  f". Wasserstein distance: distance {d}")

                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("left")
                ax.grid(axis='both',  linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)        
                ax.set_ylabel('Reward in ticks')
                ax.set_xlabel(strat)
                ax.legend(['GP1', '', 'GP2', ''], 
                        handlelength=0.2, framealpha=0.2, loc='best', ncol=2)
            ax1.fill_between(test_x.numpy(), lower1.detach().numpy(), upper1.detach().numpy(), alpha=0.4, color='k')
            ax1.fill_between(test_x.numpy(), lower2.detach().numpy(), upper2.detach().numpy(), alpha=0.4, color='darkred')
            plt.tight_layout()
            plt.show()

        return (d > self.b)
    
    def change_point_lr(self, strat):
        
        # Compute corresponding type 2 error
        # Compute Likelihood ratio test
        # If ratio test big enough reinitialize
        gp = self.strat_gp_dict[strat]
        train_x, train_y = gp.model.train_inputs[0], gp.model.train_targets
        if gp.model.train_inputs[0].shape[0] < self.size_window:
            return False

        #train_x, test_x = gp.model.train_inputs[0][:self.p], gp.model.train_inputs[0][self.p:self.P]
        S_1, S_2 = train_x[(-self.size_window):(- self.size_window//2)], train_x[(-self.size_window//2):]
        y_1, y_2 = train_y[(-self.size_window):(- self.size_window//2)], train_y[(-self.size_window//2):]

        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(S_1, y_1)
        likelihood, model = gp.model.likelihood, gp.model
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_tilde = model(S_2)
            observed_pred = likelihood(latent_f_tilde)
        mu_tilde, K_tilde = observed_pred.mean, observed_pred.covariance_matrix
        v_h0 = K_tilde + (likelihood.noise**2)*torch.eye(self.size_window//2)


        # Compute K** mu** from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(train_x, train_y)

        likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        model = ExactGPModel(torch.zeros((0, 1), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), likelihood)
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_star = model(S_2)
            observed_pred = likelihood(latent_f_star)
        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        mu, K = observed_pred.mean, observed_pred.covariance_matrix
        v_h1 = K + (likelihood.noise**2)*torch.eye(self.size_window//2)


        # Compute delta Vh0 Vh1
        v_h0_inv = torch.inverse(v_h0)
        v_h1_inv = torch.inverse(v_h1)
        delta = v_h1_inv - v_h0_inv
        target_bis = y_2 - mu_tilde

        R = -y_2 @ v_h1_inv @ y_2 + target_bis @ v_h0_inv @ target_bis - torch.log(torch.det(v_h1_inv)) + torch.log(torch.det(v_h0_inv))
        
        # If threshold is specified, abide by it, otherwise try to compute one
        if self.delta:
            return R >= self.delta
        else:
            # Conpute Threshold based on type 1 error
            aux = v_h0 @ delta
            mu_h0 = v_h0.shape[0] - mu_tilde @ v_h1_inv @ mu_tilde - torch.trace( v_h1_inv @ v_h0)

            sum_lambda_0 = torch.trace(aux @ aux)
            largest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_window//2 - 1, self.size_window//2 - 1))
            smallest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
            largest_eih0 = max(np.absolute(largest_eih0), np.absolute(smallest_eih0))

            tau_I = mu_h0 + max(torch.sqrt(-8*np.log(self.type_1_error)*(sum_lambda_0 + mu_tilde @ v_h1_inv @ v_h0 @ v_h1_inv @ mu_tilde)), torch.tensor(-8*np.log(self.type_1_error)*largest_eih0))

            #In case significant test compute equivalent second threshold
            aux = v_h1 @ delta
            #mu_h1 = -torch.trace(aux)
            mu_h1 = -v_h1.shape[0] + mu_tilde @ v_h0_inv @ mu_tilde + torch.trace( v_h0_inv @ v_h1)
            sum_lambda_1 = torch.trace(aux @ aux)
            largest_eih1 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_window//2 - 1, self.size_window//2 - 1))
            smallest_eih1 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
            largest_eih1 = max(np.absolute(largest_eih1), np.absolute(smallest_eih1))
            error_II = torch.max(torch.exp(-((mu_h1 - tau_I)**2)/(8*(sum_lambda_1 + mu_tilde @ v_h0_inv @ v_h1 @ v_h0_inv @ mu_tilde))), torch.exp(-((mu_h1 - tau_I)/(8*torch.tensor(largest_eih1))))) # = lambda 2

            # If test inferior, we already know there is no change
            if R <= tau_I:
                return False, R, tau_I, error_II

            #elif error_II < self.threshold:
                #return True, R, tau_I, error_II

            else:
                return True, R, tau_I, error_II
        
    def change_point_adaga(self, strat):

        """
        This method applies ADAGA streaming GP regression.
        """
        gp = self.strat_gp_dict[strat]

        #train_x = gp.model.train_inputs[0][(-self.size_window):(- self.size_window//2)]
        train_x = gp.model.train_inputs[0][(-self.size_window):]
        train_y = gp.model.train_targets[(-self.size_window):]
        if train_y.shape[0] < self.size_window:
            return False
        
        ada = AdaptiveRegionalization_bandit(train_x, train_y, self.delta, self.size_window, n_ind_pts=10, kern="RBF")
        
        return ada.regionalize()

        

