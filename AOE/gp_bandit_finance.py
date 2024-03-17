
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
from botorch.utils.transforms import standardize, normalize, unnormalize

from .plots import rescale_plot
from scipy.linalg import eigh

from .gp_bandit import gp_bandit, ExactGPModel
from .gp_utils import Wasserstein_GP_mean

class gp_bandit_finance:
    def __init__(self, strategies, bandit_algo='TS', 
                            likelihood = gpytorch.likelihoods.GaussianLikelihood(),
                            train_x    = torch.zeros((0, 1), dtype=torch.float64),
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

        self.strategies    = strategies
        self.training_iter = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo   = bandit_algo
        self.bandit_params = bandit_params
        self.verbose       =  verbose
        self.likelihood    = likelihood
        self.records       = {} # Traces of what happens
        self.reinit        = reinit

        # optimisation of computations
        self.last_change_point_test_results = {}
        for algo_name in ('TS_NS', 'UCB_NS', 'TS_WAS', 'UCB_WAS', 'MAB_UCB', 'UCB_LR', 'RANDOM', 'GREEDY'):
            self.last_change_point_test_results[algo_name] = {}

        # TODO Check if that's alright
        #self.likelihood.noise = torch.tensor(0.0001)

        self.strat_gp_dict = {}
        for strat in self.strategies.keys():
            self.strat_gp_dict[strat] = gp_bandit(likelihood,
                                                    bandit_algo,
                                                    train_x,
                                                    train_y,
                                                    bandit_params = bandit_params,
                                                    training_iter = training_iter)
        if bandit_algo == 'TS_NS':
            self.size_buffer   = self.bandit_params['size_window']
        elif bandit_algo == 'UCB_NS':
            self.size_buffer   = self.bandit_params['size_window']
            self.lamb          = self.bandit_params['lambda']
        elif bandit_algo == 'TS_WAS':
            self.size_window   = self.bandit_params['size_window']
            self.b             = self.bandit_params['threshold'] ### Must be even to divide last points by two
            self.size_buffer   = self.bandit_params['size_buffer']
            # records
            print('INTIALIZING!')
            self.records['was_distances'] = {s: [] for s in self.strat_gp_dict.keys() }
        elif bandit_algo == 'UCB_WAS':
            self.size_window   = self.bandit_params['size_window']
            self.b             = self.bandit_params['threshold'] ### Must be even to divide last points by two
            self.lamb          = self.bandit_params['lambda']
            self.size_buffer   = self.bandit_params['size_window']
            # records
            print('INTIALIZING!')
            self.records['was_distances'] = {s: [] for s in self.strat_gp_dict.keys() }
        elif bandit_algo == 'MAB_UCB':
            self.size_window   = self.bandit_params['size_window']
            self.delta_I       = self.bandit_params['delta_I'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.strat_t       = {} ### Dictionary of time since reset for ucb calculation
            for strat in self.strategies.keys():
                self.strat_t[strat] = 0
        elif bandit_algo == 'GREEDY':
            self.size_window   = self.bandit_params['size_window']
            self.delta_I       = self.bandit_params['delta_I'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.strat_t       = {} ### Dictionary of time since reset for ucb calculation
            for strat in self.strategies.keys():
                self.strat_t[strat] = 0
        elif bandit_algo == 'MAB_TS':
            self.size_window   = self.bandit_params['size_window']
            self.delta_I       = self.bandit_params['delta_I'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']
            self.strat_t       = {} ### Dictionary of time since reset for ucb calculation
            for strat in self.strategies.keys():
                self.strat_t[strat] = 0
        elif bandit_algo == 'UCB_LR':
            self.size_window   = self.bandit_params['size_window']
            self.delta_I       = self.bandit_params['delta_I'] ### Bound on type 1 error
            self.lamb          = self.bandit_params['lambda']

            if 'check_type_II' in self.bandit_params:
                self.check_type_II = self.bandit_params['check_type_II']
            else:
                self.check_type_II = False
                
            if 'force_threshold' in self.bandit_params:
                self.force_threshold = self.bandit_params['force_threshold']
            else:
                self.force_threshold = None
            self.records['lr_statistic'] = {s: [] for s in self.strat_gp_dict.keys() }
            self.records['noise'] = {s: [] for s in self.strat_gp_dict.keys() }
            self.records['lengthscale'] = {s: [] for s in self.strat_gp_dict.keys() }

    def select_best_strategy(self, features):
        # before testing the change points show the learnt GPs
        if self.verbose: 
            print('Plotting the GPs of strategies for the bandit: ', self.bandit_algo)
            self.plot_strategies()
            plt.show()

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

        elif self.bandit_algo == 'GREEDY':
            "Compute ucb for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_greedy(strat, features)
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
            "Compute ucb for classical mab"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                self.strat_t[strat] += 1
                ucb_strat = self.compute_ucb_mab(strat)
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

        # the bandit is now eligible for change point detection
        if strat in self.last_change_point_test_results[self.bandit_algo].keys():
            if self.verbose: print('I am deleting the change point results of ', strat)
            self.last_change_point_test_results[self.bandit_algo].pop(strat, None)

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

        elif self.bandit_algo == 'MAB_UCB':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'GREEDY':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'MAB_TS':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'UCB_LR':
            model.update_data(x_new, y_new)
            # model.update_data(x_new, y_new)
        elif self.bandit_algo == 'RANDOM':
            # no need for points
            best_strat = 'OK'
        else:
            best_strat = 'NOT IMPLEMENTED'
        
        if retrain_hyperparameters:
            if self.bandit_algo not in ("RANDOM", "MAB_UCB", "MAB_TS"): #add exceptions here
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
    
    def compute_greedy(self, strat, features):
        "Compute ucb for one strat"
        if self.verbose: print('Computing UCB for strategy:', strat)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        return gp.compute_greedy(t_x)
    
    def compute_ts(self, strat, features):
        "Compute thompson sampling for one strat"
        if self.verbose: print('Computing thompson sampling for strategy:', strat)
        
        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        return gp.compute_ts(t_x)

    def compute_ts_was(self, strat, features):
        "Compute Thompson Sampling for one strat with was check first"
        if self.verbose: print('Computing TS Waserstein for strategy:', strat)   
            
        # First Compute the waser distance
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets
            
            lv = train_x.min().item()
            uv = train_x.max().item()

            if strat in self.last_change_point_test_results[self.bandit_algo].keys():
                # it means no point has been added or reint has been done, so no need to redo test
                b_changepoint, was_distance = self.last_change_point_test_results[self.bandit_algo][strat]
            else:
                b_changepoint, was_distance = self.change_point(strat, lv = lv, uv = uv)
                if self.verbose: print('I am storing the change point test results for ', strat)
                self.last_change_point_test_results[self.bandit_algo][strat] =  b_changepoint, was_distance

            # Record the distances
            self.records['was_distances'][strat] += [was_distance] 

            if b_changepoint:
                if self.verbose: 
                    print('REGIME CHANGE! Plotting the GPs of strategies for the bandit: ', self.bandit_algo)
                    self.plot_strategies()
                    plt.show()
                    
                # print('REGIME CHANGE UCB WAS !!!!!! for strategy:', strat)
                
                # posterior_mean_1, lower1, upper1, \
                #     posterior_mean_2, lower2, upper2 = \
                #         self.posterior_sliding_window_confidence(strat, n_test = 100, lv=lv, uv=uv)

                # posterior_mean_1, posterior_covar_1, \
                #     posterior_mean_2, posterior_covar_2 = \
                #         self.posterior_sliding_window_covar(strat, 100, lv, uv)

                # ### Compute wasserstein distance between gp:
                # d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), 
                #             posterior_mean_2.numpy(), posterior_covar_2.numpy())

                
                # # Plot
                # fig, ax = plt.subplots(1, 1)
                # test_x = torch.linspace(lv, uv, 100).double()
                # ax.plot(test_x.numpy(), posterior_mean_1.numpy(), 'b')
                # ax.plot(test_x.numpy(), posterior_mean_2.numpy(), 'r')
                # # Shade between the lower and upper confidence bounds
                # ax.fill_between(test_x.numpy(), lower1.detach().numpy(), upper1.detach().numpy(), alpha=0.5)
                # ax.fill_between(test_x.numpy(), lower2.detach().numpy(), upper2.detach().numpy(), alpha=0.5)
                # # legend / title
                # ax.legend(['mean 1', 'mean 2'])
                # ax.set_title(f'Bandit UCB WAS \n strat {strat} \n Was dist= {round(d, 3)} ')
                # plt.show()

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
                self.last_change_point_test_results['TS_WAS'][strat] = {}
                self.last_change_point_test_results['TS_WAS'].pop(strat, None)

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

            if strat in self.last_change_point_test_results[self.bandit_algo].keys():
                # it means no point has been added or reint has been done, so no need to redo test
                b_changepoint, was_distance = self.last_change_point_test_results[self.bandit_algo][strat]
            else:
                b_changepoint, was_distance = self.change_point(strat, lv = lv, uv = uv)
                if self.verbose: print('I am storing the change point test results for ', strat)
                self.last_change_point_test_results[self.bandit_algo][strat] =  b_changepoint, was_distance

            # Record the distances
            self.records['was_distances'][strat] += [was_distance] 

            if b_changepoint:
                if self.verbose: 
                    print('REGIME CHANGE! Plotting the GPs of strategies for the bandit: ', self.bandit_algo)
                    self.plot_strategies()
                    plt.show()
                    
                # print('REGIME CHANGE UCB WAS !!!!!! for strategy:', strat)
                
                # posterior_mean_1, lower1, upper1, \
                #     posterior_mean_2, lower2, upper2 = \
                #         self.posterior_sliding_window_confidence(strat, n_test = 100, lv=lv, uv=uv)

                # posterior_mean_1, posterior_covar_1, \
                #     posterior_mean_2, posterior_covar_2 = \
                #         self.posterior_sliding_window_covar(strat, 100, lv, uv)

                # ### Compute wasserstein distance between gp:
                # d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), 
                #             posterior_mean_2.numpy(), posterior_covar_2.numpy())

                
                # # Plot
                # fig, ax = plt.subplots(1, 1)
                # test_x = torch.linspace(lv, uv, 100).double()
                # ax.plot(test_x.numpy(), posterior_mean_1.numpy(), 'b')
                # ax.plot(test_x.numpy(), posterior_mean_2.numpy(), 'r')
                # # Shade between the lower and upper confidence bounds
                # ax.fill_between(test_x.numpy(), lower1.detach().numpy(), upper1.detach().numpy(), alpha=0.5)
                # ax.fill_between(test_x.numpy(), lower2.detach().numpy(), upper2.detach().numpy(), alpha=0.5)
                # # legend / title
                # ax.legend(['mean 1', 'mean 2'])
                # ax.set_title(f'Bandit UCB WAS \n strat {strat} \n Was dist= {round(d, 3)} ')
                # plt.show()

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
                self.last_change_point_test_results['UCB_WAS'][strat] = {}
                self.last_change_point_test_results['UCB_WAS'].pop(strat, None)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)
    

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
            #print(f"p value test:{test_kol}")
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

        # record the current lenght scale and noise values
        self.records['noise'][strat] += [self.strat_gp_dict[strat].model.likelihood.noise  ]
        self.records['lengthscale'][strat] += [ self.strat_gp_dict[strat].model.covar_module.base_kernel.lengthscale]
        
        # First Compute the LR stat
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets
            
            if strat in self.last_change_point_test_results[self.bandit_algo].keys():
                # it means no point has been added or reint has been done, so no need to redo test
                b_changepoint = self.last_change_point_test_results[self.bandit_algo][strat]
            else:
                try:
                    b_changepoint, lr_statistic, tau_I, delta_II = self.change_point_lr(strat) #, tau_I 
                    if self.verbose: print('I am storing the change point test results for ', strat, ' with LR')
                    self.last_change_point_test_results[self.bandit_algo][strat] =  b_changepoint
                    self.records['lr_statistic'][strat] += [ (lr_statistic, tau_I, delta_II) ]
                except Exception as e:
                    print('Error while compute LR statistic:', str(e))
                    b_changepoint, lr_statistic, tau_I, delta_II = False, None,  None,  None

            if b_changepoint:
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

                    self.strat_gp_dict[strat].train() # train only if there are point

                # remove last test
                self.last_change_point_test_results[self.bandit_algo][strat] = {}
                self.last_change_point_test_results[self.bandit_algo].pop(strat, None)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)



    def plot_strategies(self, strats = "all", lv=None, uv=None, n_test=100, xlabel=None):
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

    def OLD_posterior_sliding_window_confidence(self, strat, n_test = 100, lv = -1, uv = 1, training_iter=50):
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
        
        #bounds = torch.tensor([[-1.0], [1.0]])
        #train_y_normalized =  normalize(train_y.detach(), bounds=bounds)

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
        #o_gaussLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        #o_gaussLikelihood.noise = torch.tensor(0.0001)
        
        gp_1 = gp_bandit(self.likelihood,
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
        #o_gaussLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        #o_gaussLikelihood.noise = torch.tensor(0.0001)

        gp_2 = gp_bandit(self.likelihood,
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
            
            #bounds = torch.tensor([[-1.0], [1.0] ])
            #train_y_normalized =  normalize(train_y.detach(), bounds=bounds)

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

        return (d > self.b), d
    
    
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
        likelihood_gp, model = gp.model.likelihood, gp.model
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_tilde = model(S_2)
            observed_pred = likelihood_gp(latent_f_tilde)
        mu_tilde, K_tilde = observed_pred.mean, observed_pred.covariance_matrix
        v_h0 = K_tilde + (likelihood_gp.noise**2)*torch.eye(self.size_window//2)


        # Compute K** mu** from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(train_x, train_y)
        
        likelihood_alt = gpytorch.likelihoods.GaussianLikelihood(noise=0.01)
        #likelihood_alt              = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = torch.tensor([0.01]), 
        #                                                                                learn_additional_noise=False) 

        model      = ExactGPModel(torch.zeros((0, 1), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), likelihood_alt)
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_star = model(S_2)
            observed_pred = likelihood_alt(latent_f_star)

        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        mu, K = observed_pred.mean, observed_pred.covariance_matrix
        v_h1  = K + (likelihood_alt.noise**2)*torch.eye(self.size_window//2)

        # Compute delta Vh0 Vh1
        v_h0_inv = torch.inverse(v_h0)
        v_h1_inv = torch.inverse(v_h1)
        delta = v_h1_inv - v_h0_inv
        target_bis = y_2 - mu_tilde

        R = -y_2 @ v_h1_inv @ y_2 + target_bis @ v_h0_inv @ target_bis # the real LR stat is R - torch.log(torch.det(v_h1)) + torch.log(torch.det(v_h0))

        # Conpute Threshold based on type 1 error
        aux   = v_h0 @ delta
        mu_h0 = v_h0.shape[0] - mu_tilde @ v_h1_inv @ mu_tilde - torch.trace( v_h1_inv @ v_h0)

        sum_lambda_0 = torch.trace(aux @ aux)
        largest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_window//2 - 1, self.size_window//2 - 1))
        smallest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
        largest_eih0 = max(np.absolute(largest_eih0), np.absolute(smallest_eih0))

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

            #print('I m forcing the threshold to ', self.force_threshold, ' and the LR stat is ', float(R), ' and tau_I is', float(tau_I))
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
            
