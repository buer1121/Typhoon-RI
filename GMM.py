import pickle
from sklearn.mixture import GaussianMixture
import numpy as np
import scipy.stats

class GMM:
    def __init__( self, n_components=10 ):
        self.n_components = n_components
        self.gm = GaussianMixture( n_components )
    
    def fit( self, data ):
        self.gm.fit( data )
        self.means = self.gm.means_
        self.covariances = self.gm.covariances_
        self.weights = self.gm.weights_

    def _multivariate_gaussian( self, x, mu, sigma ):
        m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(sigma))
        m_dist_x = np.dot(m_dist_x, (x-mu))
        return 1-scipy.stats.chi2.cdf(m_dist_x, len( x ) )
    
    def predict_proba( self, x ):
        proba = 0
        means, covariances, weights, n_components = self.means, self.covariances, self.weights, self.n_components
        for i in range(n_components):
            proba += weights[i] * self._multivariate_gaussian( x, means[i], covariances[i] )
        return proba
    
    def save( self, path ):
        with open( path, "wb" ) as fw:
            pickle.dump({
                "n_components": self.n_components,
                "means": self.means,
                "covariances": self.covariances,
                "weights": self.weights
            }, fw)
    
    def load( self, path ):
        with open( path, "rb" ) as fr:
            ckpt = pickle.load( fr )
        self.gm = GaussianMixture( ckpt["n_components"] )
        self.means = ckpt["means"]
        self.covariances = ckpt["covariances"]
        self.weights = ckpt["weights"]