from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma
from collections import defaultdict

try:
    import cPickle as pickle
except:
    import pickle

def logsum(A, axis=None):
    """Computes the sum of A assuming A is in the log domain.
    Returns log(sum(exp(A), axis)) while minimizing the possibility of
    over/underflow.
    """
    Amax = A.max(axis)
    if axis and A.ndim > 1:
        shape = list(A.shape)
        shape[axis] = 1
        Amax.shape = shape
    Asum = np.log(np.sum(np.exp(A - Amax), axis))
    Asum += Amax.reshape(Asum.shape)
    if axis:
        # Look out for underflow.
        Asum[np.isnan(Asum)] = - np.Inf
    return Asum

def normalize(A, axis=None):
    A += np.finfo(float).eps
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum

def E_square_x(mu,sigma):
    if len(mu.shape)==1:
        mu = mu[np.newaxis, :]
        sigma = sigma[np.newaxis, :]
    square_mu = np.sum(np.square(mu), axis=1)    #u^u
    tr_sigma = np.trace(sigma, axis1=1, axis2=2)    #tr(sigma)
    return square_mu+tr_sigma

def E_squre_xminusy(mu1,sigma1,mu2,sigma2):
    x1 = E_square_x(mu1, sigma1)
    x2 = E_square_x(mu2, sigma2)
    x12 = np.dot(mu1, mu2.T)
    return x1[:, np.newaxis]+x2[np.newaxis, :]-2*x12

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log_like_Gauss(x, mu, tau):
    """
    Log probability for Gaussian with diagnal matrices.
    x_i \in N(mu, 1/tau *Identity)
    x: n*d
    mu: d
    tau: precesion parameter
    lnP = -0.5 * (ln2pi + lndet(cv) + (obs-mu)cv(obs-mu))
    """
    n, ndim = x.shape

    dln2pi = ndim * np.log(2.0 * np.pi)
    lndetv = ndim * np.log(tau)
    q = tau * np.sum(np.square((x-mu)), axis=1)
    li = 0.5*(lndetv - dln2pi - q)
    return li

def log_like_Gamma(x, alpha, beta):
    z = alpha * np.log(beta) - gammaln(alpha) + (alpha - 1)*np.log(x) - beta*x
    return z

def E_lnpi_Dirichlet(alpha):
    return digamma(alpha) - digamma(alpha.sum())

def E_ln_Gamma(alpha,beta):
    return digamma(alpha) - np.log(beta)

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    import pylab
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    pylab.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    import pylab
    reenable = False
    if pylab.isinteractive():
        pylab.ioff()
    pylab.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    pylab.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
    pylab.axis('off')
    pylab.axis('equal')
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        pylab.ion()
    pylab.show()

def recall_num(pred, score, num):
    ind = np.argsort(-pred)
    if num <= len(pred) and np.sum(score)!=0:
        res = np.sum(score[ind[:num]])/np.sum(score)
    else:
        res = 1
    return res

'''
def plot_recall(pred, score):
    n = len(pred)
    recall = []
    per = []
    for i in range(1,1000):
        per += [i/1000]
        recall += [recall_num(pred, score, round(n*i/1000))]
    plt.plot(per, recall)
    plt.show()


def pre_recall(VB, X_test, r_test, Brand_id_test, User_id_test, U, nrec):
    # nrec: # of recommendation
    recall = np.zeros(nrec)
    nu = 0
    for i in range(U):
        if len(np.where(User_id_test == i)[0]) == 0:
            continue
        nu += 1
        X = X_test[User_id_test == i, :]
        r = r_test[User_id_test == i]
        Brand_id = Brand_id_test[User_id_test == i]
        User_id = User_id_test[User_id_test == i]

        pred = VB.predict(X, Brand_id, User_id)
        recall_i = np.zeros(nrec)
        for j in range(nrec):
            recall_i[j] = recall_num(pred, r, j+1)

        recall += recall_i
    recall = recall/nu

    plt.plot(recall)
    plt.show()

    return recall
'''

from collections import defaultdict

def calculate_recall(User_id_test, r_test, pred_test, nrec=100):
    # nrec: number of recommendation

    # First map the predictions to each user.
    user_pred = defaultdict(list)
    for i in range(len(r_test)):
        user_pred[User_id_test[i]].append((pred_test[i], r_test[i]))

    recalls = defaultdict(list)
    for uid, user_ratings in user_pred.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        recall = list()
        t = sum((true_r == 1) for (_, true_r) in user_ratings)
        for k in range(nrec):
            if len(user_ratings) > k and t > 0:
                pt = sum((true_r == 1) for (_, true_r) in user_ratings[:k+1])
                recall.append(pt / t)
            else:
                recall.append(1.0)

        recalls[uid] = recall

    recall_mean = 0
    for uid, recall in recalls.items():
        recall_mean += np.array(recall)
    recall_mean = recall_mean / len(recalls)

    #plt.plot(recall_mean)
    #plt.show()

    return recalls

def calculate_precision(User_id_test, r_test, pred_test, nrec=100):
    # nrec: number of recommendation

    # First map the predictions to each user.
    user_pred = defaultdict(list)
    for i in range(len(r_test)):
        user_pred[User_id_test[i]].append((pred_test[i], r_test[i]))

    precisions = defaultdict(list)
    for uid, user_ratings in user_pred.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        precision = list()
        t = sum((true_r == 1) for (_, true_r) in user_ratings)
        for k in range(nrec):
            pt = sum((true_r == 1) for (_, true_r) in user_ratings[:k+1])
            precision.append(pt / (k+1))


        precisions[uid] = precision

    return precisions

def calculate_NDCG(User_id_test, r_test, pred_test, nrec=100):
    # nrec: number of recommendation

    # First map the predictions to each user.
    user_pred = defaultdict(list)
    for i in range(len(r_test)):
        user_pred[User_id_test[i]].append((pred_test[i], r_test[i]))

    NDCGs = defaultdict(list)
    for uid, user_ratings in user_pred.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        ndcg = list()
        t = sum((true_r == 1) for (_, true_r) in user_ratings)
        dcg=0
        idcg=0
        for k in range(nrec):
            if t == 0:
                ndcg = [1.0]*nrec
                break

            if len(user_ratings) > k:
                dcg += user_ratings[k][1]/np.log2(k+2)
                if k < t:
                    idcg += 1/np.log2(k+2)
            ndcg.append(dcg/idcg)

        NDCGs[uid] = ndcg

    ndcg_mean = 0
    for uid, ndcg in NDCGs.items():
        ndcg_mean += np.array(ndcg)
    ndcg_mean = ndcg_mean / len(NDCGs)

    #plt.plot(ndcg_mean)
    #plt.show()

    return NDCGs

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def avg_perf(performances):
    perf_mean = 0
    for uid, perf in performances.items():
        perf_mean += np.array(perf)
    perf_mean = perf_mean / len(performances)
    return perf_mean