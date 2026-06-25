import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from sklearn.model_selection import cross_val_score, train_test_split, GroupShuffleSplit
from sklearn.linear_model import HuberRegressor
from sklearn.utils import resample

import datetime
import itertools
from scipy.integrate import trapezoid
from scipy.stats import pearsonr, linregress
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.patches as patches
from matplotlib.transforms import Bbox
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import networkx as nx
#import pygraphviz as pgv

import json,ast,time


from mrmr import mrmr_classif, mrmr_regression
from copy import deepcopy


class Mrmr_gpr():
    def __init__(self, Xlims=[], length_scales=[], length_scale_bounds=[], obj_fct=object, 
                predictors_to_log=False,npts=10,ucb_fact=7,min_rel_length_scale=0.02,min_func_val=-6,use_log_obj_fct=True,
                use_huber=False, regressor='gpr'):
        self.length_scales = length_scales
        self.length_scale_bounds = length_scale_bounds
        self.noise_level_bounds = [(1e-10, 1000)]
        self.noise_level = 0.01
        self.min_rel_length_scale = min_rel_length_scale
        self.min_func_val = min_func_val
        self.kernel_type = 'RBF'
        self.kernel = [] # the user defined kernel if kernel_type=='user
        self.verbose = False
        self.npts = npts # number of points along each dimension for the approx obj fct
        self.gp = object
        self.gps_g = [] # list of trained regressors
        self.X = [] # list of predictors, for training & shuffling
        self.X_full = [] # unrestricted list to grab from
        self.X_test = [] # list of predictors from additional data, not trained & not shuffled
        self.X_m = [] # mean of each predictor
        self.X_s = [] # standard deviation of each predictor
        self.standardize=False
        self.predictor_names = [] 
        self.target_name = ''
        self.Xlims = Xlims
        self.Jac1 = []
        self.Jac2 = []
        self.Jacs = [] # list of dictionaries
        self.y = [] # list of objectives, for training & shuffling
        self.y_m = 1 # the mean of the objectives, for the y normalization
        self.y_full = [] # unrestricted list of targets
        self.y_test = [] # list of objectives from additional data, not trained & not shuffled
        self.yall = [] # array of objectives, for fit_simple_graph
        self.Xpr = []
        self.ypr = []
        self.Xmesh = []
        self.Ymesh = [] # for 2D representation
        self.ypr_std = []
        self.ucb_fact = ucb_fact # factor for sigma to control the upper confidence bound
        self.obj_fct = obj_fct # a function handle to the objective function
        self.predictors_to_log = predictors_to_log # want to convert predictors to logarithm internally?
        self.use_log_obj_fct = use_log_obj_fct
        self.use_huber = use_huber
        self.regressor = regressor # could be 'gpr'(default), 'huber' or 'user'
        self.figs = [] # container for pyplot Figure objects

        if len(length_scales) == 0: # WAS: ==1. Now it is always true
            for ii,Xlim in enumerate(self.Xlims):
                if self.predictors_to_log:
                    drange = np.log10(Xlim[1]) - np.log10(Xlim[0])
                else:
                    drange = Xlim[1] - Xlim[0]
                self.length_scales.append(drange/2)
                if len(self.length_scale_bounds) == 0:
                    self.length_scale_bounds.append((drange*self.min_rel_length_scale,drange*10))


        self.init_gpr(length_scales=self.length_scales,length_scale_bounds=self.length_scale_bounds)
        self.prepare_Xpred()

    def apply_white_blacklist(self, df, target, targets, whitelist, exclude, missing=None, whitelist_exact_match=False):

        features = [ff for ff in df.columns]
        if whitelist_exact_match:
            features = whitelist
        else:
            if whitelist: 
                features0 = [ff for ff in features if any([white in ff for white in whitelist])] # whitelist potentially interesting predictors
                features = features0

        for exclude0 in exclude:
            features = [ff for ff in features if (exclude0 not in ff)]

        features = [ff for ff in features if (ff not in targets)] # remove the target from the features
        X = df.loc[:,features].to_numpy().astype(float)
        y = df.loc[:,target].to_numpy().astype(float)
        

        self.X = X
        self.y = y

        if 'kfold_groups' in df.columns:
            self.groups = df.kfold_groups.to_numpy().astype(int) # !!!!! NEW !!!!

        self.yall = y
        if missing!=None:
            X,y = self.handle_missing_legacy(X,y,missing)
        return features, X, y


    def handle_missing(self, df, predictors, targets, predictors_whitelist, predictors_blacklist,
                        targets_whitelist, targets_blacklist, missing, handle_missing,min_var):

        """for any algorithm that needs a complete X,y matrix without missing numbers

        There is also a more efficient function handle_missing_pairs

        Args:
            df (Pandas DataFrame): All datasets in rows, predictors & targets in columns
            predictors (list of string): each entry must correspond to an entry in df.columns. Can be empty.
                                        In this case, df.columns  is assumed
            targets (list of string): each entry must correspond to an entry in df.columns. If empty:
                                        targets = predictors (symmetic matrix, use to see if predictors are iid)
            predictors_whitelist (list of string): strings that must occur in predictor to be included
            predictors_blacklist (list of string): strings that must not occur in predictor to be included
            missing (scalar): a number that should be interpreted as missing
            handle_missing (string): None: do not handle; 'keep_predictors': remove datasets with missing values,
                                        'keep_datasets': remove predictors with missing datasets
                                        default: None
            min_var (integer): minimum number of different entries for column to be included. Default
        """

        #before doing anything else: remove columns with too little variation
        Xy = df.to_numpy()
        nvars = np.zeros((Xy.shape[1],)) # number of different entries
        rdyn = np.zeros((Xy.shape[1],)) # relative dynamics
        for ii in range(Xy.shape[1]):
            rrr=Xy[:,ii]
            try:
                if (len(rrr) > 1) & (np.mean(rrr)!=0):
                    dyn = np.max(rrr)-np.min(rrr)
                    rdyn[ii] = dyn / np.mean(rrr)
                else:
                    rdyn[ii]=0
                nvars[ii] = len(set(list(rrr)))
            
            except TypeError:
                nvars[ii] = 0
        
        #print('variability of data per columns:')  # !!!! RRREMOVE !!!!!
        
        colgood = df.columns[np.logical_not(nvars < min_var)]
        nvars_good = nvars[np.logical_not(nvars < min_var)]
        
        
        #for cg, ng in zip(colgood,nvars_good): # !!!! RRREMOVE !!!!!
        #    print(cg, ng)

        if len(predictors)==0:
            features = colgood
        else:
            features = [pp for pp in predictors if pp in colgood]
                
        if predictors_whitelist: 
            features0 = [ff for ff in features if any([white in ff for white in predictors_whitelist])] # whitelist potentially interesting predictors
            features = features0

        for exclude0 in predictors_blacklist:
            features0 = [ff for ff in features if (exclude0 not in ff)]
            features = features0

        if (len(targets)==0) & (len(targets_whitelist)==0) & (len(targets_blacklist)==0):
            targets = features
        else:
            targets = [pp for pp in targets if pp in colgood] # WAS: colgood which was BUG!! repaired!
                    
            if targets_whitelist: 
                targets0 = [ff for ff in targets if any([white in ff for white in targets_whitelist])] # whitelist potentially interesting predictors
                targets = targets0

            for exclude0 in targets_blacklist:
                targets0 = [ff for ff in targets if (exclude0 not in ff)]
                targets = targets0            
        
        tf = np.array(list(set(list(targets)+list(features)))) # NEW!!!!! added the second "list"
        Xy = df[tf].to_numpy() # array containing predictors and targets

        if handle_missing == 'keep_predictors':
            bad = Xy==missing # a boolean array
            realbad = np.sum(bad,axis = 1) # datasets containing at least one missing value evaluate to True
            Xygood = Xy[np.logical_not(realbad),:]# removing all datasets with missing values
            colgood = tf

        elif handle_missing == 'keep_datasets':
            bad = Xy==missing # a boolean array
            realbad = np.sum(bad,axis = 0) # columns containing at least one missing value evaluate to True
            colgood = tf[np.logical_not(realbad)] # the remaining column names
            Xygood = Xy[:,np.logical_not(realbad)]# removing all datasets with missing values

        elif handle_missing == None:
            Xygood = Xy
            colgood = tf
        
        df1 = pd.DataFrame(Xygood,columns=colgood) # df1 contains only selected predictors & targets & no missing values

        pred_out = [ff for ff in features if ff in colgood]
        targ_out = [ff for ff in targets if ff in colgood]

        return df1, pred_out, targ_out

    def handle_missing_pairwise(self,Xy, missing = -1):
        # X and y: numpy arrays
        bad = Xy==missing # a boolean array
        realbad = np.sum(bad,axis = 1) # datasets containing at least one missing value evaluate to True
        Xygood = Xy[np.logical_not(realbad),:]# removing all datasets with missing values
        return Xygood[:,0],Xygood[:,1]




    def handle_missing_legacy(self,X,y,exclude):
        # convention: missing converted to NaN
        #X = self.X 
        #y = self.y
        Xy = np.hstack((X,y.reshape(-1,1)))
        df_Xy = pd.DataFrame(Xy)
        df_Xy.replace(to_replace = exclude, value= np.nan, inplace=True) # convefrt exclude to NaN
        df_Xy.dropna(inplace=True,how='any') # drop any row with at leat one NaN
        Xy = df_Xy.to_numpy()
        X = Xy[:,:-1]
        y = Xy[:,-1]

        #bad = Xy==exclude # a boolean array
        #badX = X==exclude # a boolean array
        #if len(badX.shape)>1: # should be matrix
        #    badXsum = np.sum(badX,axis = 1) # datasets in which at least one predictor has missing values
        #else:
        #    badXsum = badX
        #bady = y==exclude # a boolean array
        #overlap = np.sum(np.logical_not(badXsum) & np.logical_not(bady)) # as many overlapping points as possible for good prediction
        #rel_overlap = overlap / np.sum(np.logical_not(bady))
        #grow = np.sum(np.logical_not(badXsum) & bady )
        
        #realbad = np.sum(bad,axis = 1)
        #X = X[np.logical_not(realbad),:]
        #y = y[np.logical_not(realbad)]
        
        #self.X = X # !!!!!! Do we need these global variables??
        #self.y = y 
        return X,y




    def init_gpr(self, length_scales=[], length_scale_bounds=[]):
     
        if self.use_huber:
            self.gp = HuberRegressor() 
        
        elif self.regressor == 'rf':
            self.gp = RandomForestRegressor()

        elif self.regressor == 'gbr':
            self.gp = GradientBoostingRegressor(n_estimators=10)

        elif self.regressor == 'gpr':

            if self.kernel_type == 'RBF':
                kernel = 1.0*RBF(length_scale = length_scales, length_scale_bounds = length_scale_bounds) \
                + WhiteKernel(noise_level = self.noise_level, noise_level_bounds = self.noise_level_bounds)

            elif self.kernel_type == 'Matern_32':
                kernel = WhiteKernel(noise_level=self.noise_level, noise_level_bounds=self.noise_level_bounds) \
                + 0.3 * Matern(length_scale = length_scales, length_scale_bounds = length_scale_bounds, nu=1.5)
            
            elif self.kernel_type == 'user':
                kernel = self.kernel

            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
        elif self.regressor=='user':
            pass


    def prepare_Xpred(self):
        DD=[]
        for Xlim in self.Xlims:
            if self.predictors_to_log:
                DD.append(list(np.linspace(np.log10(Xlim[0]),np.log10(Xlim[1]),self.npts)))
            else:
                DD.append(list(np.linspace(Xlim[0],Xlim[1],self.npts)))

        self.Xpr = list(itertools.product(*DD, repeat=1))
        if len(DD)==2:
            self.Xmesh, self.Ymesh = np.meshgrid(DD[0],DD[1])




    def fit(self,df, features, target, normalize_y = False, method = 'split', exclude='none',test_size = 0.3, cvnum = 5,n_shuffle=20,
                                        min_var=1,min_rel_overlap=0,min_grow=0,min_pts=10,verbose=True,
                                        length_scales_start = 1, length_scale_lims = (0.01, 100)):

        self.predictor_names = features 
        self.target_name = target

        self.length_scales = [length_scales_start for _ in features]
        self.length_scale_bounds = [length_scale_lims for _ in features]
        self.X = df.loc[:,features].to_numpy().astype(float)
        self.y = df.loc[:,target].to_numpy().astype(float)
        
        
        if 'kfold_groups' in df.columns:
            self.groups = df.kfold_groups.to_numpy().astype(int) # !!!!! NEW !!!!

        self.yall = deepcopy(self.y)
        RMSE_train, RMSE_test, acc, R2 = self.fit_gpr2(method = method,n_shuffle=n_shuffle,select_predictors = [],
                                        verbose=verbose, standardize=True, normalize_y = normalize_y, test_size = test_size, cvnum = cvnum,exclude=exclude,
                                        min_var=min_var,min_rel_overlap=min_rel_overlap,
                                        min_grow=min_grow,min_pts=min_pts)
        return RMSE_train, RMSE_test, acc, R2


    def fit_gpr2(self, method = 'split', n_shuffle = 50, test_size = 0.3, select_predictors=[], verbose=False, standardize=False,
                 normalize_y = False, cvnum=5, exclude = 'none', min_var=1, min_rel_overlap = 0, min_grow = 1, min_pts= 10):
        # fit a GPR to the data using different shuffles between test and train
        # test generalization
        # plot & export a statistics over jacobians (1st and second derivatives) 
        # adapted from zzzzz_fit_gpr_xiaoyan2.py

        # method: 'split': using skl.model_selection.train_test_split()
        # 'bootstrap': using skl.utils.resample() with replacement. Will yield better estimate for standard error of mean!
        # 'GSS': Group shuffle split.

        X = np.array(self.X)
        if select_predictors == []:
            select_predictors = np.arange(X.shape[1])

        self.init_gpr(np.array(self.length_scales)[select_predictors], np.array(self.length_scale_bounds)[select_predictors])

        
        X = X[:,select_predictors]
        y = deepcopy(self.y)

        '''if exclude!='none': # For matrix completion. Assume exclude==unknown
            #print('shape before exclude',X.shape) # Only useful if all other columns are complete!
            Xy = np.hstack((X,y.reshape(-1,1)))
            bad = Xy==exclude # a boolean array
            realbad = np.sum(bad,axis = 1)
            X = X[np.logical_not(realbad),:]
            y = y[np.logical_not(realbad)]
            #print('shape after exclude',X.shape)'''

        if exclude!='none': # NEW !!!!! For matrix completion. Assume exclude==unknown
            #print('shape before exclude',X.shape) # Should work also if many columns are incomplete
            leny = len(y)
            X,y = self.handle_missing_legacy(X,y,exclude)
            rel_overlap = len(y)/leny
            grow = 1e32 # !!!!!!!!!!!!!!!!!!!! Don't know what this is good for so I set it to very high value!!!!!!
            
            #Xy = np.hstack((X,y.reshape(-1,1)))
            #bad = Xy==exclude # a boolean array
            #badX = X==exclude # a boolean array
            #if len(badX.shape)>1: # should be matrix
            #    badXsum = np.sum(badX,axis = 1) # datasets in which at least one predictor has missing values
            #else:
            #    badXsum = badX
            #bady = y==exclude # a boolean array
            #overlap = np.sum(np.logical_not(badXsum) & np.logical_not(bady)) # as many overlapping points as possible for good prediction
            #rel_overlap = overlap / np.sum(np.logical_not(bady))
            #grow = np.sum(np.logical_not(badXsum) & bady )
            
            #realbad = np.sum(bad,axis = 1)
            #X = X[np.logical_not(realbad),:]
            #y = y[np.logical_not(realbad)]

        else:
            rel_overlap = 1
            grow = 9999
            nvars = [999,999]
            rdyn = [1]

        X_orig = deepcopy(X)
        y_orig = deepcopy(y) # conserve state before standardize to include into Jacs

        p_names = np.array(self.predictor_names)[select_predictors]

        if X.shape[0] < min_pts:
            print('too few points. Skipping fit')
            print(X.shape)
            print(p_names)
            self.Jacs.append({'mark_bad':True})
            return 0,0,0,0

        if rel_overlap < min_rel_overlap:
            print('relative overlap below allowed limits. Skipping')
            self.Jacs.append({'mark_bad':True})
            return 0,0,0,0

        if grow < min_grow:
            print('Too few or no additional matches. Skipping')
            self.Jacs.append({'mark_bad':True})
            return 0,0,0,0

        # it may be that after the exclusion, some columns became invariant.
        # if this happens, do not attempt a GPR

        nvars = np.zeros((X.shape[1],)) # number of different entries
        rdyn = np.zeros((X.shape[1],)) # relative dynamics
        for ii in range(X.shape[1]):
            rrr=X[:,ii]
            if (len(rrr) > 1) & (np.mean(rrr)!=0):
                dyn = np.max(rrr)-np.min(rrr)
                rdyn[ii] = dyn / np.mean(rrr)
            else:
                rdyn[ii]=0

            nvars[ii] = len(set(list(rrr)))    


        if any(nvars < min_var): 
            print('Too little variation in column detected. Skipping fit')
            print(p_names[ii])
            self.Jacs.append({'mark_bad':True})
            return 0,0,0,0    

        if any(np.abs(rdyn) < 1e-4): 
            print('Relative dynamics <1e-4. Probably roundoff-error. Skipping fit')
            print(p_names[ii])
            self.Jacs.append({'mark_bad':True})
            return 0,0,0,0

        # TODO bring matrix completion to a new level using graphs like in Daniil Bash et al., 2021
        #print('relative overlap:', rel_overlap, 'additional points:', grow, 'variations:',nvars)
        print('relative overlap:', rel_overlap, 'variations:',nvars)
        
        if standardize:
            self.X_m = np.mean(X,axis=0)
            self.X_s = np.std(X,axis=0)
            self.standardize = True
            X-=self.X_m # subtract mean..
            X/=self.X_s # ... then compress to variance 1

            #print('standardize has been chosen. Mean and std of predictors are:')
            #print(self.X_m)
            #print(self.X_s)

        if normalize_y:
            self.y_m = np.mean(y)
            y/=self.y_m

        Jacs = np.zeros((n_shuffle, X.shape[0], X.shape[1]))
        Jacs2 = np.zeros((n_shuffle, X.shape[0], X.shape[1]))
        Jacs0 = np.zeros((n_shuffle, X.shape[0], X.shape[1]))
        RMSE_trains = []
        RMSE_tests = []
        ratios = []
        accuracys = []
        learned_kernels = []
        lmls = []
        yslfs = []
        yslfps = []
        ytrains = []
        ytests = []
        Xtrains = []
        Xtests = []
        gps = [] # a list of trained regressors


        if method=='GSS':
            n_groups = len(np.unique(self.groups))
            #print('Number of groups',n_groups)
            test_size_abs = int(np.ceil(test_size*n_groups))
            LPGO = GroupShuffleSplit(n_splits=n_shuffle,test_size=test_size_abs,)
            rrr = LPGO.split(X, y, self.groups) 
            rrr = list(rrr)
            n_shuffle = len(rrr)



        for zzz in range(n_shuffle):
            if method=='split':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None) # WAS self.X

            elif method=='bootstrap':
                
                Xy = np.hstack((X,y.reshape(-1,1)))
                data = pd.DataFrame(Xy)
                train = resample(data, replace=True, n_samples=len(data))
                test = data[~data.index.isin(train.index)]
                X_train = train.iloc[:,:-1].to_numpy() # all except last cols
                y_train = train.iloc[:,-1].to_numpy() # last column
                X_test = test.iloc[:,:-1].to_numpy()
                y_test = test.iloc[:,-1].to_numpy()

            if method=='GSS':
                i_train = rrr[zzz][0]
                i_test = rrr[zzz][1]
                X_train = X[i_train,:]
                y_train = y[i_train]
                X_test = X[i_test,:]
                y_test = y[i_test]

            self.gp.fit(X_train, y_train)
            scores = cross_val_score(self.gp, X_train, y_train, cv=cvnum)

            accuracys.append(scores.mean())
            if (self.use_huber) or (self.regressor!='gpr'):
                lmls.append(1)

            else:
                lmls.append(self.gp.log_marginal_likelihood(self.gp.kernel_.theta))

            if self.use_huber:
                yslf = self.gp.predict(X_train).reshape(-1,1) # !!!!!!!!! without the reshape THIS GAVE WRONG RMSE VALUES!!!!!!!!!!
                yslfp = self.gp.predict(X_test).reshape(-1,1)  # Now corrected

            elif self.regressor in ['rf','gbr','user']:
                yslf = self.gp.predict(X_train)
                yslfp = self.gp.predict(X_test)         

            else:
                yslf,yslf_std = self.gp.predict(X_train, return_std=True)
                yslfp,yslfp_std = self.gp.predict(X_test, return_std=True)

            yslfs.append(yslf)
            yslfps.append(yslfp)
            ytrains.append(y_train)
            ytests.append(y_test)

            Xtrains.append(X_train)
            Xtests.append(X_test)
            gps.append(deepcopy(self.gp))
            
            if (self.use_huber) or (self.regressor!='gpr'):
                learned_kernels.append(None)
            else:
                learned_kernels.append(self.gp.kernel_)

            RMSE_train = np.sqrt(np.mean((yslf - y_train)**2))
            RMSE_test = np.sqrt(np.mean((yslfp - y_test)**2))


            if verbose:
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                if (not self.use_huber) and (self.regressor=='gpr'):
                    print("\nLearned kernel: %s" % self.gp.kernel_)
                    print("Log-marginal-likelihood: %.3f"
                        % self.gp.log_marginal_likelihood(self.gp.kernel_.theta))
                print('RMSE_train =', RMSE_train)
                print('RMSE_test =', RMSE_test)

            RMSE_trains.append(RMSE_train)
            RMSE_tests.append(RMSE_test)
            ratio = RMSE_test/RMSE_train
            ratios.append(ratio)

            Jac, Jac2, Jac0 = self.jacobian(X,dxrel = 1e-3) # axis 0: all points; axis 1: all predictors

            # TODO HANDLE NORMALIZED Y HERE !!!!!!! NEW !!!!!!!

            if standardize:
                Jacs[zzz,:,:] = self.y_m * Jac / self.X_s.reshape(1,-1) 
                Jacs2[zzz,:,:] = self.y_m * Jac2 / (self.X_s.reshape(1,-1)**2) # !!!removes select_predictors because X_s is already selected
            else:
                Jacs[zzz,:,:] = self.y_m * Jac
                Jacs2[zzz,:,:] = self.y_m * Jac2
            Jacs0[zzz,:,:] = self.y_m * Jac0
        
        ratios = np.array(ratios)

        #print('ratios test/train',ratios) # !!!!!!!!!!!!!! FOR TESTING !!!!!!!!!!!

        if self.regressor in ['gpr', 'user']: # TODO CUSTOMIZE!!!!!!!
            rat_lims = (0.8, 1.2)
        elif self.regressor in ['rf', 'gbr']:
            rat_lims = (0.6, 1.6)

        good_ones = ((ratios <= rat_lims[1]) & (ratios >= rat_lims[0])) # pick only the shuffles with good generalization    

        one_offset = abs(ratios - 1) # harvest the result graph where ratio is closest to 1
        best_one = np.argmin(one_offset) 
        #best_one = np.argmax(accuracys) # alternatively: harvest the result graph which has highest score (but generalizes worse!)
        y_train_best = ytrains[best_one]
        yslf_best = yslfs[best_one]
        y_test_best = ytests[best_one]
        yslfp_best = yslfps[best_one]
        Xtrains_best = Xtrains[best_one]
        Xtests_best = Xtests[best_one]
        learned_kernels_best = learned_kernels[best_one]

        Jacs_g = Jacs[good_ones,:,:]
        if Jacs_g.shape[0] == 0:
            print('Warning! Strong over-or underfitting detected! Marked as bad')
            good_ones = ((ratios <= 10000) & (ratios >= 0.001)) # this will pick all
            mark_bad = True 
            Jacs_g = Jacs[good_ones,:,:]
            #print(ratios)
        else:
            mark_bad = False

        Jacs2_g = Jacs2[good_ones,:,:]
        Jacs0_g = Jacs0[good_ones,:,:]
        
        RMSE_trains_g = np.array(RMSE_trains)[good_ones]
        RMSE_tests_g = np.array(RMSE_tests)[good_ones]

        RMSE_tests_g*=self.y_m # in case normalize_y has been used!! NEW!!!!!!
        RMSE_trains_g*=self.y_m

        ratios_g = np.array(ratios)[good_ones]
        accuracys_g = np.array(accuracys)[good_ones]
        lmls_g = np.array(lmls)[good_ones]
        Jacs_m = np.mean(Jacs_g,axis=0)
        Jacs_s = np.std(Jacs_g,axis=0)
        Jacs2_m = np.mean(Jacs2_g,axis=0)
        Jacs2_s = np.std(Jacs2_g,axis=0)
        Jacs0_m = np.mean(Jacs0_g,axis=0)
        Jacs0_s = np.std(Jacs0_g,axis=0)

        # statistics over the abolute predictor dynamics
        DeltaPs = np.zeros((Jacs_g.shape[0],)) # we are using Jacs but we could also use Jacs0 directly...
        DeltaP_m = np.zeros((X.shape[1],))
        DeltaP_s = np.zeros((X.shape[1],))

        DeltaP0s = np.zeros((Jacs0_g.shape[0],)) # here we are using Jacs0 instead
        DeltaP0_m = np.zeros((X.shape[1],))
        DeltaP0_s = np.zeros((X.shape[1],))

        # and finally, all well generalizing regressors
        #self.gps_g = np.array(gps,dtype=object,subok=True)[good_ones]
        self.gps_g = [] # !!!!NEW !!!!! 
        gox = np.where(good_ones)[0] # a np array of the positions of the True values
        for good_one in gox:
            self.gps_g.append(gps[good_one])

        if standardize:
            Xd = X*self.X_s + self.X_m 
        else:
            Xd = X

        for ii in range(Xd.shape[1]):
            iXXX = np.argsort(Xd[:,ii]) # sort x according to current predictor
            #XXX = self.X[iXXX,ii] # Careful!! X can (but need not) be standardized, while self.X always is original
            XXX = Xd[iXXX,ii] # but due to handle_missing, X can have less rows than self.X!!!!!! possible BUG !!!!!!

            for jj in range(Jacs_g.shape[0]):
                YYY = Jacs_g[jj,iXXX,ii]
                DeltaPs[jj] = trapezoid(YYY, XXX) # yields the overall range of the target sampled by the predictor
            DeltaP_m[ii] = np.mean(DeltaPs)
            DeltaP_s[ii] = np.std(DeltaPs)
            
            for jj in range(Jacs_g.shape[0]): # NEW!!!!!!!!!!!!!!!
                YYY = Jacs0_g[jj,iXXX,ii]
                DeltaP0s[jj] = YYY[-1]-YYY[0] # yields the overall range of the target sampled by the predictor
            DeltaP0_m[ii] = np.mean(DeltaP0s)
            DeltaP0_s[ii] = np.std(DeltaP0s)

        RMSEs = RMSE_trains_g
        uy = np.std(self.y) # the baseline std of y data
        R2s = 1-(RMSEs**2)/(uy**2)
        R2s_m = np.mean(R2s)
        
        #print('RMSE_trains_g:',RMSE_trains_g) # !!!!!!!!!!!!!!!!! FOR TESTING !!!!!!!!!
        #print('Baseline std (uy)',uy)

        self.Jacs.append({'p_names': p_names, 'X':X_orig, 'y':y_orig, 'y_m':self.y_m, 'X_m': self.X_m, 'X_s': self.X_s,
                             'RMSE_train': RMSE_trains_g,'RMSE_test':RMSE_tests_g,
                            'accuracy':accuracys_g,'R2s':R2s, 'LML':lmls_g, 'Jacs0_m':Jacs0_m, 'Jacs0_s':Jacs0_s,
                            'Jacs1_m':Jacs_m, 'Jacs1_s':Jacs_s, 'Jacs2_m':Jacs2_m, 'Jacs2_s':Jacs0_s,
                            'y_train':y_train_best, 'y_train_fit':yslf_best, 'y_test':y_test_best,'y_test_fit':yslfp_best,
                            'X_train':Xtrains_best, 'X_test':Xtests_best, 'learned_kernel':learned_kernels_best,
                            'DeltaP_m':DeltaP_m, 'DeltaP_s':DeltaP_s, 'DeltaP0_m':DeltaP0_m, 'DeltaP0_s':DeltaP0_s,
                            'regressors':self.gps_g, 'mark_bad':mark_bad})

        # TODO: Excel export of Jacs and Jacs2

        G = self.Jacs[-1]
        RMSE_test = np.mean(G['RMSE_test'])
        RMSE_train = np.mean(G['RMSE_train'])
        acc = np.mean(G['accuracy'])


        return RMSE_train, RMSE_test, acc, R2s_m
    


    def jacobian(self, X, dxrel = 1e-2):
    # the jacobian at an experimental data point X (with arbitrary number of predictors)
    # dxrel: differential normalized to range of respectivve dimension
        #X = np.array(self.X)

        #Xbest = np.array(self.X[argmax(self.y)])
        Xrange = np.array([max(X[:,ii])-min(X[:,ii]) for ii in range(X.shape[1])])
        dx = Xrange* dxrel

        Jac0 = np.zeros(X.shape) # first derivative
        Jac = np.zeros(X.shape) # first derivative
        Jac2 = np.zeros(X.shape) # second derivative
        for jj in range(X.shape[0]):
            y0 = self.gp.predict(X[jj,:].reshape(1, -1))
            for ii in range(X.shape[1]):
                X1 = X[jj,:] * 1.0
                X1[ii]+=dx[ii]
                dy0 = self.gp.predict(X1.reshape(1, -1))
                X1[ii]-=2*dx[ii]
                dy1 = self.gp.predict(X1.reshape(1, -1))
                Jac[jj,ii] = (dy0-dy1)/(2*dx[ii])
                Jac2[jj,ii] = (dy0-y0)/dx[ii] - (y0-dy1)/dx[ii]
                Jac0[jj,ii] = y0
        return Jac, Jac2, Jac0
    
    def mrmr(self,df,targets,excludes,whitelist,max_pred=5,missing ='none',whitelist_exact_match=False):
        out = []
        for target,exclude in zip(targets,excludes):
            features,X,y= self.apply_white_blacklist(df,target,targets,whitelist,exclude,missing,whitelist_exact_match=whitelist_exact_match)

            YM = pd.Series(y)
            df11 = pd.DataFrame(X,columns = features)
            df11.to_excel('df11_test.xlsx')
            selected_features = mrmr_regression(X=df11, y=YM, K=max_pred) # mrmr
            out.append(selected_features)
        return out



    def plot_result_plot(self,n_jacs=0, fn = ''):
        tgn = self.target_name

        G = self.Jacs[n_jacs]

        RMSE_test = np.mean(G['RMSE_test'])
        RMSE_train = np.mean(G['RMSE_train'])
        acc = np.mean(G['accuracy'])
        R2 = np.mean(G['R2s'])


        if len(self.X_test)!=0:
            print('drin')
            X_train = self.Jacs[-1]['X_train'] # the shuffle that produced the best generalization
            y_train = self.Jacs[-1]['y_train'] # the shuffle that produced the best generalization
            self.gp.fit(X_train, y_train)

            if self.standardize:
                X_test = (self.X_test - self.X_m.reshape(1,-1))/self.X_s.reshape(1,-1)
            else:
                X_test = self.X_test
            ypred,ypred_std = self.gp.predict(X_test, return_std=True)
            ypred*=self.y_m # bring back to original scale if normalize_y has been chosen
            ypred_std*=self.y_m

        fig1 = plt.figure() # switch on for the result figure
        fig1.subplots_adjust(left=0.18,right=0.95,top=0.95,bottom=0.18)
        ax = fig1.gca()


        ytr = G['y_train'] * self.y_m # in case normalize_y has been chosen
        ytrf = G['y_train_fit'] * self.y_m 
        ytst = G['y_test'] * self.y_m
        ytstf = G['y_test_fit'] * self.y_m

        #ax.plot(G['y_train'], G['y_train_fit'],'o',color='C0',label='training') # OLD. worked
        #ax.plot(G['y_test'],G['y_test_fit'],'o',color='C1',label='test')
        
        ax.plot(ytr, ytrf,'o',color='C0',label='training') # NEW !!!!
        ax.plot(ytst,ytstf,'o',color='C1',label='test')        
        
        if len(self.X_test)!=0:
            ax.scatter(self.y_test,ypred,edgecolor='black',facecolor='white',label='extra')
        #ax.plot([min(G['y_train']),max(G['y_train'])],[min(G['y_train']),max(G['y_train'])],'-') # OLD. WORKED
        ax.plot([min(ytr),max(ytr)],[min(ytr),max(ytr)],'-') # OLD. WORKED        
        
        ax.set_xlabel('Experimental ' + tgn)
        ax.set_ylabel('Predicted ' + tgn)
        txt='RMSE (train) = %0.3f,\n RMSE (test) = %0.3f,\n score(R2) = %0.3f,\n score(R2CV) = %0.3f' %(RMSE_train,RMSE_test,R2,acc)
        ax.text(0.7, 0.2, txt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.legend()
        ax.set_title('Result Plot')
        #plt.show()
        if fn !='':
            
            with pd.ExcelWriter(fn, mode='w') as writer:
                raus = np.hstack((G['y_train'].reshape(-1,1),G['y_train_fit'].reshape(-1,1)))
                df = pd.DataFrame(raus,columns = ['ytruth','ypred'])
                df.to_excel(writer, sheet_name = 'train')
                raus = np.hstack((G['y_test'].reshape(-1,1),G['y_test_fit'].reshape(-1,1)))
                df = pd.DataFrame(raus,columns = ['ytruth','ypred'])
                df.to_excel(writer, sheet_name = 'test')
        return fig1
    def optimize(self,n_iter=20):

        if len(self.X)==0:
            print('No predictors. Proposing random predictors instead')
            X0 = []
            for Xlim in self.Xlims:
                X0.append(float(np.random.uniform(Xlim[0],Xlim[1],1)))
            self.X.append(X0)
        else:
            if self.predictors_to_log:
                self.X = list(np.log10(np.array(self.X))) # convert incoming predictors to log 

        if len(self.y)==0:
            y = self.get_y(self.X[-1])
            self.y.append(y)

        for ii in range(n_iter):
            if ii == (n_iter-1):
                self.gpr_fit(verbose=True)
            elif ii>10:
                self.gpr_fit(verbose=False)
            else:
                self.gpr_fit()

            self.Jac1.append(list(self.jacobian()[0])) # document the evolution of the Jacobian
            self.Jac2.append(list(self.jacobian()[1]))

            self.gpr_predict()
            self.get_ucb()
            
            y = self.get_y(self.max_ucb)
            if not np.isnan(y):
                self.X.append(self.max_ucb)

                self.y.append(y if y>self.min_func_val else self.min_func_val)
                print('New data:',self.max_ucb,'-log_errsq:',y)
            else:
                print('errsq = nan. Fake data will distort the approx obj fct')
                self.X.append(self.max_ucb)
                self.y.append(self.min_func_val)

    def single_point(self,Xpr,return_std=False, return_mean_error=False):
        # returns the mean of all the well performing predictors at X_pr
        # if desired, returns also the standard deviation and the error of the mean.

        if len(Xpr.shape)==1:
            if len(self.predictor_names)==1:
                Xpr = Xpr.reshape(-1,1)
            else:
                Xpr = Xpr.reshape(1,-1)
        if len(self.Jacs)>0: # if a recent fit is available
            X_m = self.Jacs[-1]['X_m']
            X_s = self.Jacs[-1]['X_s']
        else:
            X_m = self.X_m 
            X_s = self.X_s
        Xpr = (Xpr-X_m)/X_s # apply the scaling used for the trained regressor
        yprs = []
        ypr_stds = []
        for gp in self.gps_g:
            self.gp = gp
            ypr, ypr_std = self.gp.predict(Xpr, return_std=True)
            yprs.append(ypr)
            ypr_stds.append(ypr_std)

        yprs = np.array(yprs)
        ypr_stds = np.array(ypr_stds)
        ypr = np.mean(yprs,axis=0)
        ypr_std_av = np.mean(ypr_stds,axis=0)
        ypr_std2 = np.std(yprs,axis=0)

        # pick the larger one for each point
        ypr_stds = np.vstack((ypr_std_av.reshape(1,-1),ypr_std2.reshape(1,-1)))
        ypr_stds = np.sqrt(np.sum(ypr_stds**2,axis=0))

        ypr*=self.y_m
        ypr_stds*=self.y_m 
        ypr_std2*=self.y_m # !!!NEW!!!! In case, normalize_y has been chosen!

        if (not return_std) & (not return_mean_error):
            return ypr 

        else:
            out = [ypr]
            if return_std:
                out.append([ypr_stds])
            if return_mean_error:
                out.append([ypr_std2])
            return out



    def plot_partial_dependence_1D(self,fig = [],axs=None,col = 'C0',plotmask=[],fixed_axes=[]):
        # plot obj func in 1D with error bars and data points
        # like plot_partial_dependence_1D(), but allows setting the value of the non-displayed axes
        # fixed_axes: user can provide list of fixed values for the non-displayed axes
        # length must be the same as self.predictor_names


        tgn = self.target_name
        pdn = self.predictor_names

        nvar = 50 # number of points
        npanels = len(self.predictor_names) # number of predictors = numbers of panels
        nrows = int(np.ceil(npanels/2))

        # get X and y values
        if len(self.Jacs)>0: # if a recent fit is available
            X = self.Jacs[-1]['X']
            y = self.Jacs[-1]['y']
            X_m = self.Jacs[-1]['X_m']
            X_s = self.Jacs[-1]['X_s']
        else:
            X = self.X # no fit available (regressor reloaded from database)
            y = self.y
            X_m = self.X_m 
            X_s = self.X_s 

        if len(plotmask)!=X.shape[0]:
            plotmask = np.ones((X.shape[0],),dtype=bool)
        if not isinstance(axs,np.ndarray):
            fig,axs = plt.subplots(nrows = nrows,ncols=2,figsize=(9,10))
        if len(axs.shape)==1:
            axs = np.array([axs])

        for ii in range(npanels): # iterate over all predictors
            irr = int(ii/2)
            icc = np.mod(ii,2)
            axs[irr,icc].set_xlabel(pdn[ii])
            axs[irr,icc].set_ylabel(tgn)

            Xpr = np.zeros((nvar,npanels)) # features table
            for jj in range(nvar):
                # allow user to adjust values here for the other axes
                if len(fixed_axes)==len(self.predictor_names):
                    Xpr[jj,:] = fixed_axes
                else: # otherwise fix to mean
                    Xpr[jj,:] = X_m
            xstart = np.min(X[:,ii])
            xend = np.max(X[:,ii])
            val = np.linspace(xstart, xend, nvar)
            Xpr[:,ii] = val
            Xpr = (Xpr-X_m)/X_s # apply the scaling used for the trained regressor
            yprs = []
            ypr_stds = []
            for gp in self.gps_g:
                self.gp = gp
                try: # regressor might not report standard deviation
                    ypr, ypr_std = self.gp.predict(Xpr, return_std=True)
                except:
                    ypr = self.gp.predict(Xpr)
                    ypr_std = ypr*0
                yprs.append(ypr)
                ypr_stds.append(ypr_std)

            yprs = np.array(yprs)
            ypr_stds = np.array(ypr_stds)
            ypr = np.mean(yprs,axis=0)
            ypr_std_av = np.mean(ypr_stds,axis=0)
            ypr_std2 = np.std(yprs,axis=0)

            # pick the larger one for each point
            ypr_stds = np.vstack((ypr_std_av.reshape(1,-1),ypr_std2.reshape(1,-1)))
            ypr_stds = np.sqrt(np.sum(ypr_stds**2,axis=0))

            ucb = ypr.reshape(-1,)+2*ypr_stds.reshape(-1,)
            lcb = ypr.reshape(-1,)-2*ypr_stds.reshape(-1,)
            ucb2 = ypr.reshape(-1,)+2*ypr_std2.reshape(-1,)
            lcb2 = ypr.reshape(-1,)-2*ypr_std2.reshape(-1,)

            ucb*=self.y_m
            lcb*=self.y_m
            ucb2*=self.y_m
            lcb2*=self.y_m    
            ypr*=self.y_m # !!!NEW !!!! In case normalize_y has been chosen!!      

            #plt.errorbar(val.reshape(-1,),ypr.reshape(-1,),yerr=ypr_std.reshape(-1,),fmt='o')
            axs[irr,icc].plot(val.reshape(-1,),ypr.reshape(-1,),c=col)
            axs[irr,icc].fill_between(val.reshape(-1,),lcb,ucb,alpha=0.2,color=col)
            axs[irr,icc].fill_between(val.reshape(-1,),lcb2,ucb2,alpha=0.2,color=col)
            axs[irr,icc].plot(X[plotmask,ii],y[plotmask],'o',c=col)
            plt.subplots_adjust(wspace=0.3,hspace=0.35)

        #plt.show()
        return fig, axs




    def plot_predictor_pairs(self,mode='ypr',use_3D=False):
        if len(self.Jacs)>0:
            X_orig = self.Jacs[-1]['X']
            y_orig = self.Jacs[-1]['y']
        else:
            X_orig = self.X 
            y_orig = self.y


        if self.standardize:
            if len(self.Jacs)>0:
                self.X_m = self.Jacs[-1]['X_m']
                self.X_s = self.Jacs[-1]['X_s']


            X = (X_orig - self.X_m.reshape(1,-1))/self.X_s.reshape(1,-1) #WAS: self.X_m, self.X_s
            if len(self.X_test)!=0:
                X_test = (self.X_test - self.X_m.reshape(1,-1))/self.X_s.reshape(1,-1)
        else:
            X = X_orig # WAS: self.X
            if len(self.X_test)!=0:
                X_test = self.X_test
        df1 = pd.DataFrame(data=X,columns=self.predictor_names) # WAS: self.X
        if len(self.X_test)!=0:
            df_test = pd.DataFrame(data=X_test,columns=self.predictor_names) # WAS: self.X
        else:
            df_test = None

        X_train = self.Jacs[-1]['X_train'] # the shuffle that produced the best generalization
        y_train = self.Jacs[-1]['y_train'] # the shuffle that produced the best generalization
        # X_train is standardized

        self.gp.fit(X_train, y_train)

        print('plotting predictor pairs',len(self.X[0])) # WAS: self.Xpr[0]
        
        G = itertools.combinations(df1.columns,2)
        print(list(df1.columns))
        #print(list(G))
        # construction site: arrange all plots in a triangular fashion
        ncol=len(self.X[0])-1


        if ncol>1:
            fig,ax = plt.subplots(ncols=ncol,nrows=ncol, figsize=(10,10))
        else:
            fig,ax = plt.subplots(ncols=ncol,nrows=ncol, figsize=(5,5))
        

        drin = False
        for g in G:
            drin=True
            print(g)
            features_fixed = {}
            for c in df1.columns:
                if c not in g:
                    features_fixed[c] = np.mean(df1[c])
            print('features_display:',g)
            print('features_fixed:', features_fixed)

 
            xax = list(df1.columns).index(g[0])
            yax = list(df1.columns).index(g[1])-1

            if ncol>1:
                axout = ax[yax,xax]
            else:
                axout = ax



            cset0 = self.plot_predictor_pairs_core(g, features_fixed, df1, df_test, axout,fig,mode=mode,
                                                xax=xax,yax=yax,ncol=ncol,use_3D=use_3D,
                                                y = y_orig) # !!!! list() is NEW!!!!!!!!!!
        
        if ncol>1:
            plt.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.95,wspace=0.35,hspace=0.35)
        else:
            plt.subplots_adjust(left=0.25,bottom=0.2,top=0.9,right=0.95,wspace=0.35,hspace=0.35)
        

        tgn = self.target_name

        if mode=='ypr':
            fig.suptitle('Approximate objective function', fontsize=14)
            
            cbtit = tgn
        elif mode=='std':
            fig.suptitle('Uncertainty of objective function', fontsize=14)  
            cbtit = r'$\sigma$' + f'({tgn})'
        elif mode=='ucb':
            fig.suptitle('Upper confidence bound', fontsize=14)  
            cbtit =  + f'({tgn}) + 2 X ' + r'$\sigma$'

        if not use_3D:
            for ii in range(ncol): # remove axes in upper right triangle => they are empty
                for jj in range(ii+1,ncol):
                    ax[ii,jj].axis('off')
            if ncol>1:
                cb = fig.colorbar(cset0,ax=ax[ncol-2,ncol-1])
                cb.set_label(cbtit, labelpad=+1)

            else:
                cb = fig.colorbar(cset0)
                cb.set_label(cbtit, labelpad=+1)

        #plt.show()



        if not drin:
            self.plot_predictor_pairs_core( df1.columns, 0, df1, df_test) # the above generator works only for 2 or more feat.
        
        print('learned Kernel of best generalizing shuffle:')
        print(self.Jacs[-1]['learned_kernel'])
        '''
        if len(list(G))==0:
            print('YES')
            self.plot_predictor_pairs_core(df1.columns, '',df1)
        '''
        return fig, ax
    
    def plot_predictor_pairs_core(self, features_display, features_fixed, df1, df_test,ax2,fig2,
                                mode,xax=None,yax=None,ncol=None,use_3D=None,
                                y = None):



        ftd = features_display
        tgn = self.target_name
        
        if isinstance(y,np.ndarray):
            y_orig = y.reshape(-1,)
        else:
            y_orig = self.y.reshape(-1,)

        if len(self.X[0])>=2: # at least 3 predictors   WAS: self.X[0]
            #for feature in features_fixed.keys():
            #    df1 = df1.loc[lambda df1: df1[feature] == features_fixed[feature],:]

            pred1 = df1.loc[:,features_display[0]]
            pred2 = df1.loc[:,features_display[1]]

            pred1 = pred1.to_numpy()
            pred2 = pred2.to_numpy()

            if len(self.X_test)!=0:
                pred1_test = df_test.loc[:,features_display[0]]
                pred2_test = df_test.loc[:,features_display[1]]

                pred1_test = pred1_test.to_numpy()
                pred2_test = pred2_test.to_numpy()                

            else:
                pass # this has been swicthed off!!!!!!!!!!!!!!!!!!!!!!
            if use_3D: # !!!!!!!!!!!!!!!!!!!!!!!!!! REINTRODUCED!!!!!!!!!!!!!!!!!!!!!
                fig2 = plt.figure(figsize=(6.4,4.8),dpi=180)
                fig2.subplots_adjust(left=0.16,right=0.92,top=0.95,bottom=0.18)
                ax2 = fig2.gca(projection='3d') # 3D representation

            
            dyn1 = max(pred1) - min(pred1) # add some headroom to the 3d representation
            dyn2 = max(pred2) - min(pred2) # add some headroom to the 3d representation
            X1pred = np.linspace(min(pred1)-0.2*dyn1,max(pred1)+0.2*dyn1,self.npts)
            X2pred = np.linspace(min(pred2)-0.2*dyn2,max(pred2)+0.2*dyn2,self.npts)
            if self.standardize:
                ipred1 = self.predictor_names.index(features_display[0])
                ipred2 = self.predictor_names.index(features_display[1])
                X1pred_plot = (X1pred * self.X_s[ipred1]) + self.X_m[ipred1]
                X2pred_plot = (X2pred * self.X_s[ipred2]) + self.X_m[ipred2] 
                pred1_plot = (pred1 * self.X_s[ipred1]) + self.X_m[ipred1]
                pred2_plot = (pred2 * self.X_s[ipred2]) + self.X_m[ipred2] 
                if len(self.X_test)!=0:
                    pred1_plot_test = (pred1_test * self.X_s[ipred1]) + self.X_m[ipred1]
                    pred2_plot_test = (pred2_test * self.X_s[ipred2]) + self.X_m[ipred2]                     
            else:
                X1pred_plot = X1pred
                X2pred_plot = X2pred
                pred1_plot = pred1
                pred2_plot = pred2
                if len(self.X_test)!=0:
                    pred1_plot_test = pred1_test
                    pred2_plot_test = pred2_test  
            Xpred = []      
            dX1pred = X1pred[1]-X1pred[0]
            dX2pred = X2pred[1]-X2pred[0]

            Xm, Ym = np.meshgrid(X1pred_plot, X2pred_plot)

            for X1 in X1pred:
                for X2 in X2pred:
                    X0 = []
                    for feature in self.predictor_names: # this is the order in which the GPR was trained
                        if feature in features_display:
                            if features_display.index(feature)==0:
                                X0.append(X1)
                            else:
                                X0.append(X2)
                        elif feature in features_fixed.keys():
                            X0.append(features_fixed[feature])

                    Xpred.append(X0)
            if self.use_huber or (self.regressor!='gpr'):
                yout = self.gp.predict(Xpred)
                yout = yout.reshape(self.npts,self.npts)
                yout*=self.y_m # !!!!NEW!!!! In case normalize_y has ben chosen!!

            else:
                ypred,ypred_std = self.gp.predict(Xpred, return_std=True)
                #self.Xpr = Xpred # !!!!!!!!!!!!!!!!!!! Careful!! Might kill subsequent stuff! Xpred is dimensionality reduced!!
                self.ypr = ypred 
                self.ypr_std = ypred_std
                #ypred = ypred.reshape(self.npts,self.npts)
                if mode == 'ypr':
                    yout = ypred.reshape(self.npts,self.npts)
                elif mode == 'std':
                    yout = ypred_std.reshape(self.npts,self.npts)
                elif mode == 'ucb':
                    print(self.ypr.shape, self.ypr_std.shape, ypred.shape)
                    yout = self.ypr + self.ucb_fact * self.ypr_std.reshape(-1,1)
                    yout = yout.reshape(self.npts,self.npts)

                yout*=self.y_m # !!!!NEW!!!! In case normalize_y has ben chosen!!

                
            if use_3D:
                cset = ax2.plot_surface(Xm, Ym, yout.T,cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.4,
                                                    vmin=np.amin(self.y.reshape(-1,)),vmax=np.amax(self.y.reshape(-1,)))
                ax2.scatter(pred1_plot,pred2_plot,y_orig,c=y_orig,cmap=cm.coolwarm,alpha=1) #WAS coolwarm
                if self.X_test!=[]:
                    ax2.scatter(pred1_plot_test,pred2_plot_test,self.y_test.reshape(-1,), facecolor='white', edgecolor = 'gray', alpha=1)
            


            
            else:
                if mode=='ypr':
                    vmin = np.min(y_orig)
                    vmax = np.max(y_orig)
                elif mode=='std':
                    vmin = np.min(yout.reshape(-1,))
                    vmax = np.max(yout.reshape(-1,))
                elif mode=='ucb':
                    vmin = np.min(yout.reshape(-1,))
                    vmax = np.max(yout.reshape(-1,))
                


                csetf=ax2.contourf(Xm, Ym, yout.T,40, zdir='z', offset=0.00,alpha=0.85,cmap=cm.viridis,vmin=vmin,vmax=vmax) # use this one for flat contours
                cset=ax2.scatter(pred1_plot,pred2_plot,c=y_orig,edgecolor = 'black',cmap=cm.viridis,alpha=0.85) #WAS viridis alpha=1
                if self.X_test!=[]:
                    for x,y,z in zip(pred1_plot_test,pred2_plot_test,y_orig):
                        ax2.scatter(x,y, facecolor='white', edgecolor = 'black', alpha=1,
                                    vmin=np.amin(y_orig),vmax=np.amax(y_orig))           


            Nlab = len(ax2.xaxis.get_ticklabels())
            if Nlab > 3:
                fact = int(np.floor(Nlab/3))
                for label in ax2.xaxis.get_ticklabels()[::2]: # hide every other label
                    label.set_visible(False)
            Nlab = len(ax2.yaxis.get_ticklabels())
            if Nlab > 3:
                fact = int(np.floor(Nlab/3))
                for label in ax2.yaxis.get_ticklabels()[::2]: # hide every other label
                    label.set_visible(False)
            if yax==ncol-1:
                ax2.set_xlabel(ftd[0],labelpad = 12) # WAS: features_display instead of ftd
            if xax==0:
                ax2.set_ylabel(ftd[1],labelpad = 12) # NEW!!!!!

            if use_3D:
                ax2.set_xlabel(ftd[0],labelpad = 12) # WAS: features_display[0]
                ax2.set_ylabel(ftd[1],labelpad = 12) # NEW!!!!!
                if len(ax2.zaxis.get_ticklabels())>5:
                    for label in ax2.zaxis.get_ticklabels()[::2]: # hide every other label
                        label.set_visible(False)

                ax2.set_zlabel(tgn,labelpad = 12) # WAS: self.target_name
                fig2.colorbar(cset)
            else:
                if ncol>5:
                    for label in ax2.yaxis.get_ticklabels(): # hide every other label
                        label.set_visible(False)  
                    for label in ax2.xaxis.get_ticklabels(): # hide every other label
                        label.set_visible(False)     
            if mode=='ypr':
                return cset
            elif mode=='std':
                return csetf
            else:
                return csetf
                                   

        elif len(self.X[0])==-1: # two predictors # WAS: self.X[0]
            fig2 = plt.figure()
            if use_3D:
                ax2 = fig2.gca(projection='3d')
            else:
                ax2 = fig2.gca()

            pred1 = df1.loc[:,features_display[0]]
            pred2 = df1.loc[:,features_display[1]]

            pred1 = pred1.to_numpy()
            pred2 = pred2.to_numpy()

            dyn1 = max(pred1) - min(pred1) # add some headroom to the 3d representation
            dyn2 = max(pred2) - min(pred2)

            X1pred = np.linspace(min(pred1)-0.2*dyn1,max(pred1)+0.2*dyn1,self.npts)
            X2pred = np.linspace(min(pred2)-0.2*dyn2,max(pred2)+0.2*dyn2,self.npts)    
            Xpred = []
            for X1 in X1pred:
                for X2 in X2pred:
                    Xpred.append([X1,X2])

            ypred,ypred_std = self.gp.predict(Xpred, return_std=True)
            ypred = ypred.reshape(self.npts,self.npts)

            ypred*=self.y_m # !!!!NEW!!!! In case normalize_y has ben chosen!!

            #print(X1pred)
            #print(X2pred)
            Xm, Ym = np.meshgrid(X1pred, X2pred)
            #surf = ax2.plot_surface(Xm, Ym, ypred.T, cmap = cm.coolwarm,
            #                   linewidth=0, antialiased=False)
            #cont = ax2.contour(X1pred, X2pred, ypred.T,40)

            cset=ax2.contourf(X1pred, X2pred, ypred.T,40, offset=-0.02)

            fig2.colorbar(cset, shrink=0.5, aspect=5)
            #ax2.colorbar()

            #ax2.plot(X[:,0],X[:,1],y.reshape(-1,),'o',color='C0')
            ax2.set_xlabel(ftd[0])
            ax2.set_ylabel(ftd[1]) # WASW: features_display[1]

            #ax2.set_zlabel(target[0])

        elif len(self.X[0])==1: # one predictor # WAS: self.Xpr[0]

            fig2 = plt.figure()
            ax2 = fig2.gca()

            pred1 = df1.loc[:,features_display[0]]
            dyn1 = max(pred1) - min(pred1) # add some headroom to the 3d representation
            Xpred = np.linspace(min(pred1)-0.2*dyn1,max(pred1)+0.2*dyn1,self.npts)


            ypred,ypred_std = self.gp.predict(Xpred.reshape(-1,1), return_std=True)
            
            ypred*=self.y_m # !!!!NEW!!!! In case normalize_y has ben chosen!!

            ax2.plot(Xpred,ypred)
            ax2.plot(self.X,self.y,'o',color='C0')
            ax2.set_xlabel(ftd[0]) # WAS: features_display[0]
            #ax2.set_ylabel(self.target[0])

            #ax2.set_title(gp.kernel_)
            plt.show()


    def fit_gpr_autoselect(self, improvement_fact = 0.95, method='split', n_shuffle = 10, select_method = 'FOM', weight_r2 = 1,
                                    verbose = False, standardize = True, normalize_y=False, test_size = 0.3, cvnum = 5,
                                    exclude = 'None',min_var=1,max_pred=5,min_rel_overlap = 0,min_grow = 1,
                                    min_pts=10, publishable_names = False, graph = None,
                                    required = []):
        
        self.Jacs = [] # !!!!!!!!!!!!!!!!! TEST !!!!!!!!!!!!!!!!!!!!!!! 

        # find the best predictors automatically, using improvement of RMSE as indicator
        # a new predictor should improve RMSE substantially, otherwise it will just cause noise
        bestvalue = 1e33 # starting value
        ic0 = [] # list of best new predictors
        mRMSE = [] # list of mean rmse's (just accumulating)
        
        accs = []
        ics = [] # list of corresponding predictor names

        FOMs = []
        R2ss = [] # total explanation of variance
        dR2s = [] # additional explanation of variance
        dR2s_s = [] # and its variance
        preds = [] # list of predictor names
        count = 0

        uy = np.std(self.y) # the baseline std of y data
        rmse_prev = np.std(self.y) # the previously best rmse (initialized to baseline rmse)
        R2prev = 0 # the previously achieved explanation of variance

        for req in required: # add the required predictors to start with
            ic0.append(self.predictor_names.index(req))
        
        predn = self.predictor_names
        targn = self.target_name


        
        gax = graph
        #gax = pgv.AGraph(strict=False, directed=True) # same, using pygraphviz
        gax.add_node(self.target_name,rmse = uy,label=targn) # add target and baseline uncertainty
        
        
        for _ in range(len(self.predictor_names)):    

            ccc = [ii for ii,pn in enumerate(self.predictor_names) if ii not in ic0]
            for c in ccc:
                ic = ic0+[c]

                RMSE_train, RMSE_test, acc, R2 = self.fit_gpr2(select_predictors = ic,standardize=standardize,
                                normalize_y = normalize_y, verbose=verbose, method = method,
                                n_shuffle=n_shuffle, test_size = test_size, cvnum=cvnum,exclude=exclude,
                                min_var=min_var,min_rel_overlap=min_rel_overlap,min_grow=min_grow,min_pts=min_pts)

                if self.Jacs[-1]['mark_bad']:
                    RMSE_test=9999
                    RMSE_train=9999
                    acc = -9999
                    RMSEs = np.array([9999])
                else:
                    # get all well performing RMSE_train from the latest run: in self.Jacs[-1]
                    RMSEs = self.Jacs[-1]['RMSE_train']                    


                accs.append(acc)
                ics.append(ic)
                preds.append(list(np.array(predn)[ic]))



                R2s = 1-(RMSEs**2)/(uy**2)
                R2s_m = np.mean(R2s)
                R2ss.append(R2s_m)
                dR2 = R2s - R2prev
                dR2_m = np.mean(dR2)
                dR2_s = np.std(dR2)
                dR2s.append(dR2_m)
                dR2s_s.append(dR2_s)

                if select_method=='RMSE':
                    FOM = RMSE_train
                elif select_method=='FOM':
                    FOM = RMSE_test * (1-acc)
                elif select_method=='dR2':
                    FOM = 1-R2s_m # must be minimal!
                elif select_method=='weighted':
                    FOM = RMSE_test * (1/weight_r2-acc)
                mRMSE.append(RMSE_test)
                FOMs.append(FOM)

                count+=1
                if acc!=-9999:
                    print('GPR fit for :', np.array(self.predictor_names)[ic],
                         'RMSEtr/tst = %0.4f,  %0.4f, cvscore(R2) = %0.4f, R2 = %0.4f, dR2= %0.4f ' %(RMSE_train,RMSE_test,acc,R2s_m,dR2_m))
                
            
            ibest0 = np.argmin(FOMs) # WAS: mRMSE

            if select_method == 'dR2':
                crit = bestvalue - improvement_fact
            else:
                crit = bestvalue * improvement_fact
            
            if (FOMs[ibest0] < (crit)) and (len(ic0) < (max_pred-1)):
                bestvalue = FOMs[ibest0]
                ibest = ibest0
                #print('#jacs',len(self.Jacs),'ibest',ibest)
                
                best_new_pred = self.Jacs[ibest]['p_names'][-1]
                RMSEs_best = self.Jacs[ibest]['RMSE_train']
                print('Best new predictor:',best_new_pred, ',std(y)= %0.4f, RMSE = %0.4f, score(R2) = %0.4f' % (uy,mRMSE[ibest],accs[ibest]))

                prednr = best_new_pred
                gax.add_node(best_new_pred,rmse = mRMSE[ibest],label=prednr)

                #gax.add_edge(best_new_pred, self.target_name, weight = rmse_prev-gax.nodes[best_new_pred]['rmse']) # old version: weight is improvement of RMSE
                gax.add_edge(best_new_pred, self.target_name, weight = float(f'{dR2s[ibest]:.2f}')) # new version: differential explanation of variance
                #rmse_prev = gax.nodes[prednr]['rmse']

            elif (FOMs[ibest0] < (crit)) and (len(ic0) == (max_pred-1)):
                bestvalue = FOMs[ibest0]
                ibest = ibest0
                #print('#jacs',len(self.Jacs),'ibest',ibest)
                
                best_new_pred = self.Jacs[ibest]['p_names'][-1]
                RMSEs_best = self.Jacs[ibest]['RMSE_train']
                print('Best new predictor:',best_new_pred, ',std(y)= %0.4f, RMSE = %0.4f, score(R2) = %0.4f' % (uy,mRMSE[ibest],accs[ibest]))
                print('Maximum number of predictors reached.')


                
                prednr = best_new_pred
                #weight = float(f'{dR2s[ibest]:.2f}')
                
                gax.add_node(best_new_pred,rmse = mRMSE[ibest],label=prednr)

                gax.add_edge(best_new_pred, self.target_name, weight = float(f'{dR2s[ibest]:.2f}'))
                #rmse_prev = gax.nodes[prednr]['rmse']     
                break           

            else:
                print('Adding further predictors does not improve the RMSE.')
                print('Best predictors:',self.Jacs[ibest]['p_names'],', std(y)= %0.4f, RMSE = %0.4f, score(R2) = %0.4f' % (uy,mRMSE[ibest],accs[ibest]))

                break
            #R2prev = 1-(rmse_prev**2)/(uy**2) # BUG!!! This is doing the mean BEFORE the division; mean should come after!
            R2sprev = 1-(RMSEs_best**2)/(uy**2)
            R2prev = np.mean(R2sprev)
            ic0.append(self.predictor_names.index(best_new_pred))

        explanations = {'baseline':uy,'ics':ics,'predictors':preds,'mRMSE':mRMSE, 'dR2s':dR2s, 'dR2s_s':dR2s_s, 'icbest':ics[ibest]}

        gax.nodes[self.target_name]['R2']=R2ss[ibest] # add R2 to node to store the predictive power of the predictive model
        return {'best_pred':self.Jacs[ibest]['p_names'], 'RMSE': mRMSE[ibest],'R2': accs[ibest],'graph':gax,'explanations':explanations}
        
        #self.plot_jacs_2d([0])
        
        '''
        self.plot_jacs_1d(deriv=1,n_jacs=ibest) # plot first derivatives of obj funcs
        self.plot_jacs_1d(deriv=0,n_jacs=ibest) # plot obj func values at position of data points 
        self.plot_result_plot(n_jacs=ibest) # plot predicted against ground truth for each data point
        self.plot_pred_rmse(mRMSE,ics,ic0) # plot contribution of each predictor to RMSE
        '''
    
    

    def _h_mrmr(self,df, targets, excludes, whitelist=[], max_pred_mrmr = 5, max_pred_gpr = 3, improvement_fact = 0.05,
                graph = None, use_gpr = True, regressor_params={}, proxy_params = {}, whitelist_exact_match = False,
                required = []):
        # core function for the h_mrmr_gpr workflow

        print('h_mrmr for targets:',targets)
        missing = proxy_params['missing']
        best_preds = self.mrmr(df, targets, excludes, whitelist, max_pred = max_pred_mrmr,
                               missing=None,whitelist_exact_match=whitelist_exact_match)
        print(best_preds)

        if use_gpr:
            length_scale_lims = regressor_params['length_scale_lims']
            length_scales_start = regressor_params['length_scales_start']

            select_method = proxy_params['select_method'] if 'select_method' in proxy_params.keys() else 'dR2'
            min_rel_overlap = proxy_params['min_rel_overlap'] if 'min_rel_overlap' in proxy_params.keys() else 0.6
            n_shuffle = proxy_params['n_shuffle'] if 'n_shuffle' in proxy_params.keys() else 20
            normalize_y = proxy_params['normalize_y'] if 'normalize_y' in proxy_params.keys() else True
            method = proxy_params['method'] if 'method' in proxy_params.keys() else 'split'
            test_size = proxy_params['test_size'] if 'test_size' in proxy_params.keys() else 0.25
            cvnum = proxy_params['cvnum'] if 'cvnum' in proxy_params.keys() else 3

            excludes = [[] for _ in range(len(targets))]
            print('refining upstream mRMR by embedded mRMR-GPR')
            print('targets:',targets)
            best_preds, rs = self.find_proxy(df,targets,excludes=excludes, whitelists = best_preds, length_scales_start = length_scales_start,
                                            length_scale_lims = length_scale_lims, min_var = 2, max_pred = max_pred_gpr, missing = missing,
                                            min_rel_overlap = min_rel_overlap, min_grow=0,min_pts=2,select_method = select_method, graph=graph,
                                            n_shuffle = n_shuffle, improvement_fact = improvement_fact, method = method, test_size = test_size, cvnum = cvnum,
                                            required = required, normalize_y = normalize_y, whitelist_exact_match = True)
            
        else: # rely on plain mrmr and build the graph from that

            if graph==None:
                graph = nx.DiGraph() # a directed graph instance for the additional explanations (using networkx)
            for target,bpr in zip(targets,best_preds):
                if use_gpr:
                    bprx = bpr # GPR: use all predictors
                else:
                    bprx = bpr[:2] # upstream mrmr: use only the strongest 2
                graph.add_node(target,label=target)
                for best_pred in bprx:
                    graph.add_node(best_pred,label=best_pred)
                    graph.add_edge(best_pred, target, weight = 1)

            rs = []
            rs.append({'graph':graph})


        #features = list(best_preds[0])
        if graph==None:
            graph = rs[-1]['graph']
            
        for target,bpr in zip(targets,best_preds):

            if use_gpr:
                bprx = bpr # GPR: use all predictors
            else:
                bprx = bpr[:2] # upstream mrmr: use only the strongest 2

            r = self.fit(df,bprx,target,exclude=missing,min_var=1,min_rel_overlap=0,min_grow=0,min_pts=10, n_shuffle=n_shuffle,verbose=False,
                        length_scales_start=length_scales_start, length_scale_lims=length_scale_lims,normalize_y=normalize_y)

            dpar,dpar_s = self.get_feature_importance(n_jacs=-1, p_names=bprx)
            print('Feature importance:',bprx,dpar)

            # set the edge weights according to dpar

            print('setting edges to feature importance data:', bprx)
            for ii,best_pred in enumerate(bprx):
                de = graph.get_edge_data(best_pred,target)
                de['feature_importance'] =  float(f'{dpar[ii]:.4f}')
                de['weight'] = float(f'{dpar[ii]:.4f}')

        return best_preds, rs


    def h_mrmr(self, df, targets, excludes, whitelist_S=[], whitelist_P=[], 
                max_pred_mrmr = 5, max_pred_gpr = 3, n_add_layers = 0, improvement_fact = 0.05,
                method = 'split', test_size = 0.25, cvnum = 3,
                add_P = True, graph = None, use_gpr = True, regressor_params = {},
                proxy_params = {}, whitelist_exact_match = False, required = []):
        
        # hierarchical mrmr
        # build a knowledge graph where each node is the target of a predictive model
        # and the in-edges are the predictors

        excludes = [excludes[0] for _ in targets]

        best_preds, rs = self._h_mrmr(df, targets, excludes, whitelist_S, max_pred_mrmr = max_pred_mrmr, improvement_fact=improvement_fact,
                                      max_pred_gpr = max_pred_gpr, graph=None, use_gpr = use_gpr, regressor_params = regressor_params,
                                      proxy_params = proxy_params, whitelist_exact_match = whitelist_exact_match, required = required)
        gax = rs[-1]['graph']

        # Now I want to do mrmr with the nodes with no incoming edges
        for ii in range(n_add_layers):
            gax = rs[-1]['graph']
            lefts = [node for node, in_degree in gax.in_degree() if in_degree == 0] # no incoming edges - these are the new targets
            exx = excludes[0]+[n for n in gax.nodes]
            excludes = [list(set(exx)) for _ in lefts]# do not re-use nodes (acyclic directed graph)
            best_preds, rs = self._h_mrmr(df, lefts, excludes, whitelist_S, max_pred_mrmr = max_pred_mrmr,improvement_fact=improvement_fact,
                                          max_pred_gpr=max_pred_gpr,graph=gax,use_gpr=use_gpr, regressor_params = regressor_params,
                                          proxy_params = proxy_params, whitelist_exact_match = whitelist_exact_match, required = required)

            gax = rs[-1]['graph']

        if add_P:   
            gax = rs[-1]['graph']
            lefts = [node for node, in_degree in gax.in_degree() if in_degree == 0] # no incoming edges - these are the new targets
            exx = excludes[0]+[n for n in gax.nodes]
            excludes = [list(set(exx)) for _ in lefts]# do not re-use nodes (acyclic directed graph)
            best_preds, rs = self._h_mrmr(df, lefts, excludes, whitelist_P, max_pred_mrmr = max_pred_mrmr,improvement_fact=improvement_fact,
                                          max_pred_gpr = max_pred_gpr, graph=gax, use_gpr = use_gpr, regressor_params = regressor_params,
                                          proxy_params = proxy_params, whitelist_exact_match = whitelist_exact_match, required = required)

            gax = rs[-1]['graph'] 
        return gax
    
    
    def mrmr(self,df,targets,excludes,whitelist,max_pred=5,missing ='none',whitelist_exact_match=False):
        out = []
        for target,exclude in zip(targets,excludes):
            features,X,y= self.apply_white_blacklist(df,target,targets,whitelist,exclude,missing,whitelist_exact_match=whitelist_exact_match)

            YM = pd.Series(y)
            df11 = pd.DataFrame(X,columns = features)
            df11.to_excel('df11_test.xlsx')
            selected_features = mrmr_regression(X=df11, y=YM, K=max_pred) # mrmr
            out.append(selected_features)
        return out


    def find_proxy(self,df,targets,excludes,whitelists,length_scales_start,length_scale_lims, normalize_y=False,
                    min_var=1,max_pred=5,missing ='none',min_rel_overlap=0.5,min_grow=2,min_pts=10,
                    n_shuffle=10,improvement_fact =0.99,method = 'split', test_size=0.2, cvnum=2, select_method='FOM',weight_r2 = 1,
                    plot_explanations=True, publishable_names=False,graph=None, required = [],whitelist_exact_match=False):
        best_preds = []
        rs = []
        if len(excludes)!=len(targets):
            print('Warning! len(excludes)!=len(targets)')
        if len(whitelists)!=len(targets):
            print('Warning! len(whitelists)!=len(targets)')


        for target,exclude,whitelist in zip(targets,excludes,whitelists):
            features,X,y = self.apply_white_blacklist(df, target, targets, whitelist, exclude, missing=missing,
                                                           whitelist_exact_match=whitelist_exact_match)

            
            self.predictor_names = features 
            self.target_name = target

            self.length_scales = [length_scales_start for _ in features]
            self.length_scale_bounds = [length_scale_lims for _ in features]

            #print('total length of features',len(features))
            #print(features)

            if graph==None:
                graph = nx.DiGraph() # a directed graph instance for the additional explanations (using networkx)

            r = self.fit_gpr_autoselect(improvement_fact = improvement_fact,n_shuffle=n_shuffle, select_method = select_method,
                                                verbose=False, standardize=True, normalize_y = normalize_y,method=method, test_size = test_size, cvnum=cvnum,
                                                exclude = missing,min_var=min_var,max_pred=max_pred,
                                                min_rel_overlap=min_rel_overlap,min_grow=min_grow,min_pts=min_pts,
                                                weight_r2=weight_r2,publishable_names = publishable_names,graph=graph,
                                                required=required)
            best_preds.append(r['best_pred'])
            rs.append(r)
            #rout =deepcopy(r)
            #rout.pop('explanations')
            #print(rout)

            graphlist = [[v for k,v in r.items() if k not in ['graph','explanations']]]
            out_cols = ['predictors', 'RMSE_train', 'acc']
            print(graphlist)
            
            target_out = target.split(' ')[0]
            gj = nx.node_link_data(r['graph']) # convert graph to a dict that can be json serialized
            with open('graph_' + target_out + '.json', 'w') as fp:
                json.dump(gj, fp)
            

            plt.figure() # !!!!NEW !!!!!!!!!!!!!!
            
            self.graph_draw_mpl(r['graph'])# this is a pygraphviz graph!!

            expl = r['explanations']
            expl_out = [[ip,p,r,dr2,dr2_s] for ip,p,r,dr2,dr2_s in zip(expl['ics'],expl['predictors'],expl['mRMSE'],expl['dR2s'],expl['dR2s_s'])]
            
            
            with pd.ExcelWriter('graph_list_' + target_out + '.xlsx', mode='w') as writer:
                dfx = pd.DataFrame(graphlist,columns=out_cols)
                dfx.to_excel(writer, sheet_name = 'Graphs')
                dfx1 = pd.DataFrame(expl_out,columns=['Pred_indices','Predictors','mRMSE','dR2','dR2_s'])
                dfx1.to_excel(writer, sheet_name = 'Explanations')
            
            if plot_explanations:
                fig = plt.figure()
                ibest = r['explanations']['icbest'] # get the indices to the best predictors
                irest = [ii for ii in range(len(self.predictor_names))if ii not in ibest]
                iX = ibest + irest # ordered predictor indices for X axis
                ipreds = list(dfx1.Pred_indices)
                dR2s = dfx1.dR2.to_numpy()
                dR2s_s = dfx1.dR2_s.to_numpy()
                shapes = ['o','s','d','v','^']

                for ii,iX0 in enumerate(iX): # iterate over all predictors on X axis
                    for jj in range(1,len(ipreds[-1])+1): # iterate over all predictor set lengths
                        ipr = [ii0 for ii0,ip in enumerate(ipreds) if ( len(ip)==jj) & (ip[-1]==iX0)] # pointer to row index
                        if len(ipr)==1:
                            Y = dR2s[ipr[0]]
                            Y_s = dR2s_s[ipr[0]]
                            plt.plot(ii,Y,shapes[jj-1],c='C'+str(jj-1))
                            plt.errorbar(ii,Y,yerr=Y_s,c='C'+str(jj-1))
                labels = list(np.array(self.predictor_names)[iX])

                
                labx = labels
                targ2 = target

                plt.xticks(range(len(iX)), labx, rotation='vertical')
                plt.ylabel(r'$\Delta R^{2}$' + f'({targ2})')
                plt.plot(np.linspace(0,len(iX)),0*np.linspace(0,len(iX)),linewidth=0.5,c='gray')
                plt.ylim([-0.2,1])
                plt.title('Differential explanation of variance')
                #plt.show()
                self.figs.append(plt.gcf())


                fig = plt.figure() # a simpler figure for the main manuscript
                ibest = r['explanations']['icbest'] # get the indices to the best predictors
                irest = [ii for ii in range(len(self.predictor_names))if ii not in ibest]
                iX = ibest + irest # ordered predictor indices for X axis
                ipreds = list(dfx1.Pred_indices)
                dR2s = dfx1.dR2.to_numpy()
                dR2s_s = dfx1.dR2_s.to_numpy()
                shapes = ['o','s','d','v','^']


                for ii,ibest0 in enumerate(ibest): # iterate over all predictors on X axis
                    ipr = [ii0 for ii0,ip in enumerate(ipreds) if ( len(ip)==(ii+1)) & (ip[-1]==ibest0)] # pointer to row index
                    if len(ipr)==1:
                        Y = dR2s[ipr[0]]
                        Y_s = dR2s_s[ipr[0]]
                        plt.bar(ii,Y)
                        plt.errorbar(ii,Y,yerr=Y_s,c='black')
                labels = list(np.array(self.predictor_names)[ibest])


                labx = labels
                targ2 = target

                plt.xticks(range(len(ibest)), labx, rotation='vertical')
                plt.ylabel(r'$\Delta R^{2}$' + f'({targ2})')
                plt.plot(np.linspace(0,len(ibest)),0*np.linspace(0,len(ibest)),linewidth=0.5,c='gray')
                plt.ylim([-0.1,1])
                plt.title('Differential explanation of variance')
                #plt.show()
                self.figs.append(plt.gcf())            


        best_preds = [list(rr) for rr in best_preds]
        return best_preds, rs

    def graph_draw_mpl(self,G):
        nodesize=1200
        pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes

        # nodes
        nlist = [u for (u, d) in G.nodes(data=True)]
        options = {"edgecolors": "black", "node_size": nodesize, "alpha": 0.9}
        nx.draw_networkx_nodes(G, pos, nodelist=nlist, node_color="tab:blue", **options)
        labels = nx.get_node_attributes(G,'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=16, font_family="sans-serif",font_color="whitesmoke")

        # edges
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > -33]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]
        nx.draw_networkx_edges(G, pos, node_size=nodesize, edgelist=elarge, width=3)
        #nx.draw_networkx_edges(G, pos, node_size=nodesize, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")

        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size=16)

        plt.tight_layout()
        plt.axis("off")
        plt.show()
    
    def graph_update(self,dg=None,df=None,target=None):
        # Create a graph g (or update an existing one)
        # with the info provided in dataframe df

        if dg==None:
            dg = nx.DiGraph() # a directed graph instance
            dg.add_node(target,dR2 = 0) # add target and baseline uncertainty


        # analyse information in df
        df['Nps'] = [len(pp) for pp in df.Predictors] # add number of predictors
        maxn = np.max(df.Nps) # find maximum number of predictors
        cols = list(df.columns)
        # build graph
        for ii in range(1,maxn):
            df1 = df[df.Nps==ii]
            if len(df1)!=0:
                ibest = np.argmax(df1.dR2)
                dR2 = df1.iloc[ibest,cols.index('dR2')]
                predP = df1.iloc[ibest,cols.index('Predictors')] # a string as it comes from Pandas
                predL = predP[1:-1].split(',')
                predLR = predL[-1][1:]
                pred = rf'{predLR}'
                print(predLR)
                if dR2 > 0:
                    dg.add_node(pred,dR2 = 0)
                    dg.add_edge(pred, target, weight = dR2)
        
        return dg

    def get_feature_importance(self, n_jacs=0, p_names=[]):
        G = self.Jacs[n_jacs]
        dpar = [] # container for the feature importance
        dpar_s = [] # and the standard deviation thereof
        for ii in range(len(p_names)): # iterate over all predictors
            dpar.append(G['DeltaP_m'][ii]) 
            dpar_s.append(G['DeltaP_s'][ii])
        return dpar, dpar_s

    def graph_draw_mplx(self, G, fixed=None, pos=None, figsize = (10,10), gencolors=None):
        nodesize=8000

        fig = plt.figure(figsize=figsize) # NEW!!!!

        if fixed==None: # No positions provided, need to calculate them here

            fixed = []
            pos = {}
            tg = list(nx.topological_generations(G))
            stg = len(tg) - 1
            xposs = np.linspace(0,1,len(tg))
            for ii,gen in enumerate(tg):
                if ii==stg:
                    pred_type='T' # this generation refers to the target
                elif ii==0:
                    pred_type='P'# this generation refers to processing
                else:
                    pred_type='S'# rest of the generations refers to structural features
                yposs = np.linspace(0,1,len(gen))
                for jj, n in enumerate(gen):
                    fixed.append(n)
                    pos[n] = (xposs[ii],yposs[jj])
                    if gencolors!=None:
                        G.nodes[n]['color'] = gencolors[pred_type]

        pos = nx.spring_layout(G, seed=None, weight=None,fixed=fixed,pos=pos)  # positions for all nodes

        # nodes
        nlist = [u for (u, d) in G.nodes(data=True)]
        # !!! below making sure there is no div by zero
        # !!! rule may fail if n.lims[1]==0 !!!!!!!

        #colors = ["tab:blue" if ((abs(n.std/(n.val+n.lims[1]*1e-10)))<1)or(n.std==0) else "tab:red" for n in nlist]
        #options = {"edgecolors": "black", "node_size": nodesize, "alpha": 0.9}

        R2s = np.array([d['R2'] if 'R2' in d.keys() else 0 for (u,d) in G.nodes(data=True)]) # the explanation of variance by the predictive model
        linewidths = R2s*6

        print('linewidths',linewidths)

        #colors = ["tab:blue" if True else "tab:red" for n in nlist]
        nodecolors = [d['color'] for (u, d) in G.nodes(data=True)]
        options = {"edgecolors": "black", "node_size": nodesize, "alpha": 0.9, 'linewidths':linewidths}   
        
        nx.draw_networkx_nodes(G, pos, nodelist=nlist, node_color=nodecolors, **options)
        labels = nx.get_node_attributes(G,'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_family="sans-serif",font_color="whitesmoke",font_weight="bold")

        # edges
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > -33]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]
        # edge colors
        edgelist = []
        cols = []
        for node in G.nodes:
            dpars = []
            for edge in G.in_edges(node,data=True):
                dpars.append(edge[2]['feature_importance'])
            if len(dpars)>0:
                dpars = np.array(dpars)
                vmaxa = np.max(np.abs(dpars))
                vmin = -vmaxa
                vmax = vmaxa
                spread = vmax-vmin
                for edge in G.in_edges(node,data=True):
                    edgelist.append(edge)
                    cols.append(mpl.colormaps['coolwarm']((edge[2]['feature_importance']-vmin)/spread))
                    print('Edge:',edge)

        nx.draw_networkx_edges(G, pos, node_size=nodesize, edgelist=edgelist, width=6,edge_color=cols)
        #nx.draw_networkx_edges(G, pos, node_size=nodesize, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")

        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size=12)
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        plt.tight_layout()
        plt.axis("off")
        plt.show()

