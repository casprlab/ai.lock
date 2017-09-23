import numpy as np
import os
from sklearn.decomposition import PCA, IncrementalPCA


class TransformImagesPCA(object):
    'Transform images to latent space using a trained model.'
    def __init__(self,  n_components=150):
        self.pca = IncrementalPCA(n_components=n_components, batch_size=10000)

    def learn_pcs(self, X_train):
        n = X_train.shape[0]
        chunk_size = 3000
        num_partitions = int(n / chunk_size)
        remaining = n - (num_partitions * chunk_size)
        for i in range(0, n // chunk_size):
            self.pca.partial_fit(X_train[i * chunk_size: (i + 1) * chunk_size])
        if remaining > 0:
            self.pca.partial_fit(X_train[num_partitions * chunk_size:])

    def transform(self, X_test, store=False, tansformed_test_file = ""):
        X_test_pca = self.pca.transform(X_test)
        if(store and tansformed_test_file != ""):
            if not os.path.exists(tansformed_test_file):
                np.savetxt(tansformed_test_file, X_test_pca)
                print "Transformed testing data saved."
        return X_test_pca
