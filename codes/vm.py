import numpy as np
from __future__ import division
from utilfun import *
from data_process import *
from scipy.special import gammaln

class VBms:
    """
        Model with Varational Bayesian (VB) Learning.

        Observation:
        X: [N,ndim] apparel feature
        r: [N,1] whether click the recommended apparel
        User_id:  [N,1] the user id for each apparel
        Brand_id: [N,1] the brand id for each apparel

        Attributes:
          _nstyles [int] number of style
          _nuser [int]
          _nbrand [omt]
          _ndim [int] dimension of x

          ----------------prior-----------
          _theta0 [S]
          _alpha0 [int]
          _beta0 [int]

          --------------posterior-----------
          _alpha_u [int] , _beta_u [int]
          _alpha_b [int] , _beta_b [int]
          _alpha_s [int] , _beta_s [int]
          _alpha_w [int] , _beta_w [int]
          _w_mu [d] _w_sigma [d*d]
          _ws_mu [S, d] _ws_sigma [S, d*d]
          _wb_mu [B ,d] _wb_sigma [B, d*d]
          _wu_mu [U ,d] _wb_sigma [U, d*d]
          _theta [S]

          pi [S]
          z [B, S]
          u [B, S]
          tau_u [int] expection
          tau_b [int] expection
          tau_s [int] expection
          tau_w [int] expection
    """

    def __init__(self,d,U,B,S=10,theta0=0.5,alpha0=1,beta0=1):
        self._ndim = d
        self._nuser = U
        self._nbrand = B
        self._nstyle = S
        self._theta0 = np.ones(S) * theta0
        self._alpha0 = alpha0
        self._beta0 = beta0

    # ------- posteiror
        self._alpha_u = self._alpha0; self._beta_u = self._beta0
        self._alpha_b = self._alpha0; self._beta_b = self._beta0
        self._alpha_s = self._alpha0; self._beta_s = self._beta0
        self._alpha_w = self._alpha0; self._beta_w = self._beta0

        self.tau_u = self._alpha_u/self._beta_u  # expectation
        self.tau_b = self._alpha_b/self._beta_b  # expectation
        self.tau_s = self._alpha_s/self._beta_s  # expectation
        self.tau_w = self._alpha_w/self._beta_w  # expectation

        self._w_mu = np.zeros(self._ndim);   self._w_sigma = 1/self.tau_w*np.identity(self._ndim)
        self._ws_mu = np.zeros([self._nstyle, self._ndim])
        self._ws_sigma = np.asarray([(1/x*np.identity(self._ndim)) for x in self._nstyle*[self.tau_s]])
        self._wb_mu = np.zeros([self._nbrand, self._ndim])
        self._wb_sigma = np.asarray([(1 / x * np.identity(self._ndim)) for x in self._nbrand * [self.tau_b]])
        self._wu_mu = np.zeros([self._nuser, self._ndim])
        self._wu_sigma = np.asarray([(1 / x * np.identity(self._ndim)) for x in self._nuser * [self.tau_u]])


        self._theta = self._theta0 + np.random.random(self._nstyle)
        self.pi = self._theta/self._theta.sum()

        self.lik = []

    def _init_bound(self, X, r):
        nobs, ndim = X.shape
        self.r = r
        self.xi = np.sqrt(2*np.sum(np.square(X), axis=1))
        self.Lambda = 1/(2*self.xi)*(sigmoid(self.xi)-0.5)
        self.h = np.zeros(nobs)

    def _E_step(self):
        Expe = self.tau_b*E_squre_xminusy(self._wb_mu, self._wb_sigma, self._ws_mu, self._ws_sigma)
        p = E_lnpi_Dirichlet(self._theta) + self._ndim/2 * E_ln_Gamma(self._alpha_b, self._beta_b) - \
            self._ndim/2*np.log(2*np.pi)-1/2*Expe
        self.u = np.exp(p-logsum(p, 1)[:, np.newaxis])

    def _update_multinomial_parameters(self):
        self._theta = self._theta0+np.sum(self.u, axis=0)
        self.pi = self._theta/self._theta.sum()

    def _update_user_parameters(self,  X, r, User_id, Brand_id):
        for ku in range(self._nuser):
            if len(np.where(User_id == ku)[0]) == 0:
                continue
            X_ku = X[User_id == ku, :]
            lambda_ku = self.Lambda[User_id == ku]
            Brand_id_ku = Brand_id[User_id == ku]
            E_wb_ku = np.asarray([self._wb_mu[i, :] for i in Brand_id_ku])
            r_ku = r[User_id == ku]

            lambda_X = np.multiply(np.sqrt(lambda_ku[:, np.newaxis]), X_ku)
            pre_ku = self.tau_u*np.identity(self._ndim) + 2*np.dot(lambda_X.T, lambda_X)
            self._wu_sigma[ku, :, :] = np.linalg.inv(pre_ku)

            A_ku = r_ku - 0.5 - 2 * np.multiply(lambda_ku, np.sum(np.multiply(X_ku, E_wb_ku), axis=1))
            A2_ku = np.dot(A_ku, X_ku)
            self._wu_mu[ku, :] = np.linalg.solve(pre_ku, A2_ku)

    def _update_brand_parameters(self,  X, r, User_id, Brand_id):
        for kb in range(self._nbrand):
            if len(np.where(Brand_id == kb)[0]) == 0:
                continue
            X_kb = X[Brand_id == kb, :]
            lambda_kb = self.Lambda[Brand_id == kb]
            User_id_kb = User_id[Brand_id == kb]
            E_wu_kb = np.asarray([self._wu_mu[i, :] for i in User_id_kb])
            r_kb = r[Brand_id == kb]

            lambda_X = np.multiply(np.sqrt(lambda_kb[:, np.newaxis]), X_kb)
            pre_1 = 2*np.dot(lambda_X.T, lambda_X)
            pre_2 = self.tau_b*np.sum(self.u[kb, :])*np.identity(self._ndim)
            pre_kb = pre_2 + pre_1
            self._wb_sigma[kb, :, :] = np.linalg.inv(pre_kb)

            A_kb = r_kb - 0.5 - 2 * np.multiply(lambda_kb, np.sum(np.multiply(X_kb, E_wu_kb), axis=1))
            A2_kb = np.dot(A_kb, X_kb)
            A3_kb = self.tau_b*np.dot(self.u[kb, :], self._ws_mu)
            self._wb_mu[kb, :] = np.linalg.solve(pre_kb, (A3_kb+A2_kb))

    def _update_style_parameters(self):
        for ks in range(self._nstyle):
            pre_ks = self.tau_s + self.tau_b*np.sum(self.u[:, ks])
            self._ws_sigma[ks, :, :] = np.identity(self._ndim)/pre_ks

            A1_ks = self.tau_s * self._w_mu
            A2_ks = self.tau_b*np.dot(self.u[:, ks], self._wb_mu)
            self._ws_mu[ks, :] = np.dot(self._ws_sigma[ks, :, :], (A1_ks+A2_ks))

    def _update_general_parameters(self):
        self._w_sigma = np.identity(self._ndim)/(self.tau_w+self._nstyle*self.tau_s)

        a = self.tau_s*np.sum(self._ws_mu, axis=0)
        self._w_mu = np.dot(self._w_sigma, a)

    def _update_precision_parameters(self):
        self._alpha_u = self._alpha0 + self._ndim*self._nuser/2
        self._beta_u = self._beta0 + 0.5*np.sum(E_square_x(self._wu_mu, self._wu_sigma))
        self.tau_u = self._alpha_u/self._beta_u

        self._alpha_b = self._alpha0 + self._ndim*self._nbrand/2
        self._beta_b = self._beta0 + 0.5*np.sum(E_square_x(self._wb_mu, self._wb_sigma))
        self.tau_b = self._alpha_b/self._beta_b

        self._alpha_s = self._alpha0 + self._ndim*self._nstyle/2
        self._beta_s = self._beta0 + 0.5*np.sum(E_square_x(self._ws_mu, self._ws_sigma))
        self.tau_s = self._alpha_s/self._beta_s

        self._alpha_w = self._alpha0 + self._ndim/2
        self._beta_s = self._beta0 + 0.5*np.sum(E_square_x(self._w_mu, self._w_sigma))
        self.tau_w = self._alpha_w/self._beta_w

    def _update_variational_parameters(self, X, User_id, Brand_id):
        nobs, ndim = X.shape
        for i in range(nobs):
            x = X[i, :]
            wb_mu = self._wb_mu[Brand_id[i], :]
            wb_sigma = self._wb_sigma[Brand_id[i], :, :]
            wu_mu = self._wu_mu[User_id[i], :]
            wu_sigma = self._wu_sigma[User_id[i], :]
            xi2 = np.square(np.dot(x, (wb_mu+wu_mu)))+np.dot(np.dot((wb_sigma + wu_sigma), x), x)
            self.xi[i] = np.sqrt(xi2)
            self.Lambda[i] = 1 / (2 * self.xi[i]) * (sigmoid(self.xi[i]) - 0.5)
            self.h[i] = np.dot(x, (wb_mu + wu_mu))

    def _update_parameters(self, X, r, User_id, Brand_id):
        #update for the multinomial parameter
        self._update_multinomial_parameters()

        #update for the user parameters
        self._update_user_parameters(X, r, User_id, Brand_id)

        # update for the brand parameters
        self._update_brand_parameters(X, r, User_id, Brand_id)

        # update for the style parameters
        self._update_style_parameters()

        # update for the general parameters
        self._update_general_parameters()

        # update for the precision parameters:
        self._update_precision_parameters()

        # update for the variational parameters:
        self._update_variational_parameters(X, User_id, Brand_id)

    def Likelihood(self):
        l1 = np.log(sigmoid(self.xi))+np.multiply(self.r, self.h) - 0.5*(self.h + self.xi) - \
              np.multiply(self.Lambda, (np.square(self.h) - np.square(self.xi)))
        l1 = np.sum(l1)

        l2 = np.sum(log_like_Gauss(self._wu_mu, np.zeros(self._ndim), self.tau_u))

        l3 = 0
        for k in range(self._nstyle):
            l3_k = np.log(self.pi[k]) + log_like_Gauss(self._wb_mu, self._ws_mu[k, :], self.tau_b)
            l3 += np.dot(self.u[:, k], l3_k)

        l4 = np.sum(log_like_Gauss(self._ws_mu, self._w_mu, self.tau_s))
        l5 = log_like_Gauss(self._w_mu[np.newaxis, :], 0, self.tau_w)

        l6 = gammaln(self._theta.sum()) - gammaln(self._theta).sum() + np.dot((self._theta-1), np.log(self.pi))

        l7 = log_like_Gamma(self.tau_u, self._alpha_u, self._beta_u) + \
            log_like_Gamma(self.tau_b, self._alpha_b, self._beta_b) + \
            log_like_Gamma(self.tau_s, self._alpha_s, self._beta_s) + \
            log_like_Gamma(self.tau_w, self._alpha_w, self._beta_w)

        return l1+l2+l3+l4+l5+l6+l7

    def fit(self, X, r, User_id, Brand_id, niter=10000, eps=1.0e-1, ifreq=10):
        """
        Fit model parameters via Variational Bayes algorithm
        input
          X,r, User_id,Brand_id : observed data
          niter [int] : maximum number of iteration cyles
          eps [float] : convergence threshold
          ifreq [int] : frequency of printing fitting process
        """
        self._init_bound(X, r)

        L = -1.0e50
        for i in range(niter):
            self._E_step()
            self._update_parameters(X, r, User_id, Brand_id)

            L_new = self.Likelihood()

            dL = L_new - L
            self.lik += [float(L_new)]

            if abs(dL) < eps:
                self.bpar = self.u.argmax(1)
                print("%8dth iter, Likelihood = %12.6e, dL = %12.6e" % (i, L_new, 5))
                print("%12.6e < %12.6e Converged" % (dL, eps))
                break
            # print iteration info
            if i % ifreq == 0 and dL > 0.0:
                print("%8dth iter, Likelihood = %12.6e, dL = %12.6e" %(i,L_new,dL))
            elif dL < 0.0:
                print("%8dth iter, Likelihood = %12.6e, dL = %12.6e warning" %(i,L_new,dL))

            L = L_new

    def predict(self, X, brand_id, user_id):
        """
        :param X:  [M, ndim]
        :param brand_id: [M ,1]
        :param user_id:  [M ,1]
        :return: [M, 1] the probability for each recommendation
        """
        M, ndim = X.shape
        wb_mu = np.asarray([self._wb_mu[i, :] for i in brand_id])
        #wb_sigma = np.asarray([self._wb_sigma[i, :, :] for i in brand_id])
        wu_mu = np.asarray([self._wu_mu[i, :] for i in user_id])
        #wu_sigma = np.asarray([self._wu_sigma[i, :, :] for i in user_id])

        mu = np.sum(np.multiply(X, (wb_mu+wu_mu)), axis=1)
        var = np.zeros(M)
        for i in range(M):
            sigma = self._wb_sigma[brand_id[i],:,:] + self._wu_sigma[user_id[i],:,:]
            x = X[i,: ]
            var[i] = np.dot(np.dot(x, sigma), x)

        pre = sigmoid(mu/np.sqrt(1+np.pi*var/8))
        return pre

    def predict2(self, X, brand_id, user_id):
        """
        :param X:  [M, ndim]
        :param brand_id: [M ,1]
        :param user_id:  [M ,1]
        :return: [M, 1] the probability for each recommendation
        """

        pre = []
        M, ndim = X.shape
        for i in range(M):
            x = X[i, :]
            pred = 0
            for ks in range(self._nstyle):
                mu = np.dot(x, self._ws_mu[ks, :])
                var = np.sum(np.square(x)) / (self.tau_u+self.tau_b)
                pred += self.u[brand_id[i], ks] * sigmoid(mu/np.sqrt(1+np.pi*var/8))
            pre += [pred]

        return np.array(pre)

    def predict3(self, X, brand_id, user_id):
        """
        :param X:  [M, ndim]
        :param brand_id: [M ,1]
        :param user_id:  [M ,1]
        :return: [M, 1] the probability for each recommendation
        """
        M, ndim = X.shape
        BtoS = self.u.argmax(1)
        wb_mu = np.asarray([self._ws_mu[BtoS[i], :] for i in brand_id])

        mu = np.sum(np.multiply(X, wb_mu), axis=1)
        var = np.sum(np.square(X), axis=1) / (self.tau_u+self.tau_b)

        pre = sigmoid(mu/np.sqrt(1+np.pi*var/8))
        return pre

    def showModel(self,show_mu=False,show_cv=False,min_pi=0.0001):
        """
        Obtain model parameters for relavent clusters
        input
          show_mu [bool] : if print mean vectors
          show_cv [bool] : if print covariance matrices
          min_pi [float] : components whose pi < min_pi will be excluded
        output
          relavent_clusters [list of list] : a list of list whose fist index is
          the cluster and the second is properties of each cluster.
            - relavent_clusters[i][0] = mixing coefficients
            - relavent_clusters[i][1] = cluster id
            - relavent_clusters[i][2] = mean vector
            - relavent_clusters[i][3] = covariance matrix
          Clusters are sorted in descending order along their mixing coeffcients
        """
        # make a tuple of properties and sort its member by mixing coefficients
        params = sorted(zip(self.pi,range(self._nstyle),self._ws_mu,self._ws_sigma),\
            key=lambda x:x[0],reverse=True)

        relavent_clusters = []
        for k in range(self._nstyle):
            # exclude clusters whose pi < min_pi
            if params[k][0] < min_pi:
                break

            relavent_clusters.append(params[k])
            print("\n%dth component, pi = %8.3g" % (k,params[k][0]))
            print("cluster id =", params[k][1])
            if show_mu:
                print("mu =",params[k][2])
            if show_cv:
                print("cv =",params[k][3])

        return relavent_clusters

    def decode(self,eps=0.01):
        """
        Return most probable cluster ids.
        Clusters are sorted along the mixing coefficients
        """
        # take argmax
        codes = self.u.argmax(1)
        # get sorted ids
        params = self.showModel(min_pi=eps)
        # assign each observation to corresponding cluster
        clust_pos = []
        for p in params:
            clust_pos.append(codes==p[1])
        return clust_pos

    def plot1d(self,d1=0,eps=0.0001,clust_pos=None):
        """
        plot data of each cluster along one axis
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          d1 [int, optional] : id of axis
          clust_pos [list, optional] : decoded cluster postion
        """
        # plotting symbols
        symbs = ".hd^x+"
        # plot range
        l = np.arange(self._nbrand)
        # decode observed data
        if clust_pos == None:
            clust_pos = self.decode(eps)
        # import pyplot
        try :
            import matplotlib.pyplot as plt
        except ImportError :
            print("cannot import pyplot")
            return
        # plot data
        for k,pos in enumerate(clust_pos):
            symb = symbs[k // 7]
            plt.plot(l[pos], self.u[pos,d1],symb,label="%3dth cluster"%k)
        plt.legend(loc=0)
        plt.show()

    def plot2d(self,d1=0,d2=1,eps=0.01,clust_pos=None):
        """
        plot data of each cluster along two axes
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          d1 [int, optional] : id of the 1st axis
          d2 [int, optional] : id of the 2nd axis
          clust_pos [list, optional] : decoded cluster postion
        """
        symbs = ".hd^x+"
        if clust_pos == None:
            clust_pos = self.decode(eps)
        try :
            import matplotlib.pyplot as plt
        except ImportError :
            print("cannot import pyplot")
            return
        for k,pos in enumerate(clust_pos):
            symb = symbs[k // 7]
            plt.plot(self.u[pos,d1],self.u[pos,d2],symb,label="%3dth cluster"%k)
        plt.legend(loc=0)
        plt.show()

if __name__ == "__main__":
    file =  './train_data_new'
    """
    feature information, 
      fi=0: popularity
      fi=1: cid + popularity
      fi=2: cid + popularity + title
      fi=3: cid + popularity + title + attributes
    """
    M=200
    X, r, User_id, Brand_id, Sku_id, d, U, B, N = obtain_data(file, brand_min=50)
    #X_train, r_train, User_id_train, Brand_id_train, Sku_id_train, X_test, r_test, User_id_test, Brand_id_test, \
    #    Sku_id_test = sample_train_data(X, r, User_id, Brand_id, Sku_id, U, prop=0.8)
    X_train, r_train, User_id_train, Brand_id_train, Sku_id_train, X_test, r_test, User_id_test, Brand_id_test, \
           Sku_id_test = sample_data_out(X, r, User_id, Brand_id, Sku_id, U, prop=0.7)

    #X_train, r_train, User_id_train, Brand_id_train, Sku_id_train, X_test, r_test, User_id_test, Brand_id_test, \
    #Sku_id_test = splitData(X, r, User_id, Brand_id, Sku_id, prop=0.8)
    print(X_train.shape, r_train.shape, User_id_train.shape, Brand_id_train.shape)

    d = [220]
    for fi in range(1):
        X_train_fi = X_train[:, :d[fi]]
        X_test_fi = X_test[:, :d[fi]]

        print(X_train_fi.shape, r_train.shape, User_id_train.shape, Brand_id_train.shape)
        VB = VBms(d[fi], U, B, S=5)
        VB.fit(X_train_fi, r_train, User_id_train, Brand_id_train, niter=10000, eps=1.0e-2)

        ### prediction
        pred_test = VB.predict(X_test_fi, Brand_id_test, User_id_test)

        recalls = calculate_recall(User_id_test, r_test, pred_test, nrec=M)
        NDCGs = calculate_NDCG(User_id_test, r_test, pred_test, nrec=M)
        precisions = calculate_precision(User_id_test, r_test, pred_test, nrec=M)

        obj = [VB, recalls, NDCGs, precisions]

        filename = 'VB_pred_out2'+str(fi)+'.pkl'
        save_object(obj, filename)

