import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA

try:
    from umap import UMAP
except ImportError:
    def UMAP(*args, **kwargs):
        raise RuntimeError("Support for UMAP was not installed! Install with: pip install {}[{}]".format(
            __name__.split('.')[0], "umap"
        ))

from .base import Processor, Param


class Embedding(Processor):
    kwargs = Param(dict, {})


class EigenDecomposition(Embedding):
    """Eigenvalue Decomposition

    """
    n_eigval = Param(int, 32)
    which = Param(str, 'LM')
    normalize = Param(bool, True)

    def function(self, data):
        """Compute spectral embedding of `data`

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            data with samples in rows

        Returns
        -------
        :obj:`numpy.ndarray`
            Eigenvalues for spectral embedding
        :obj:`numpy.ndarray`
            Spectral embedding (eigenvectors)

        Note
        ----
        We use the fact that (I-A)v = (1-λ)v and thus compute the largest eigenvalues of the identity minus the
        data and return one minus the eigenvalue.

        """
        eigval, eigvec = eigsh(data, k=self.n_eigval, which=self.which, **self.kwargs)
        eigval = 1. - eigval

        if self.normalize:
            eigvec /= np.linalg.norm(eigvec, axis=1, keepdims=True)
        return eigval, eigvec


class TSNEEmbedding(Embedding):
    """TSNE Embedding

    """
    n_components = Param(int, default=2)
    metric = Param(str, default='euclidean')

    def function(self, data):
        tsne = TSNE(n_components=self.n_components, metric=self.metric, **self.kwargs)
        emb = tsne.fit_transform(data)
        return emb


class PCAEmbedding(Embedding):
    """PCA Embedding

    """
    n_components = Param(int, default=2)
    whiten = Param(bool, default=False)

    def function(self, data):
        pca = PCA(n_components=self.n_components, whiten=self.whiten, **self.kwargs)
        emb = pca.fit_transform(data)
        return emb


class LLEEmbedding(Embedding):
    """LocallyLinearEmbedding

    """
    n_components = Param(int, default=2)
    n_neighbors = Param(int, default=5)

    def function(self, data):
        lle = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components,
                                     **self.kwargs)
        emb = lle.fit_transform(data)
        return emb


class UMAPEmbedding(Embedding):
    """UMAPEmbedding: https://umap-learn.readthedocs.io/en/latest/index.html

    """
    n_neighbors = Param(int, default=15)
    min_dist = Param(float, default=0.1)
    metric = Param(str, default='correlation')

    def function(self, data):
        umap = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric)
        emb = umap.fit_transform(data)
        return emb