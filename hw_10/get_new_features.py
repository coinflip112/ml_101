import numpy as np
from sklearn import (
    manifold,
    datasets,
    decomposition,
    ensemble,
    discriminant_analysis,
    random_projection,
    neighbors,
)

X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method="modified")
X_mlle = clf.fit_transform(X)
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
X_mds = clf.fit_transform(X)
embedder = manifold.SpectralEmbedding(
    n_components=2, random_state=0, eigen_solver="arpack"
)
X_se = embedder.fit_transform(X)
tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
X_tsne = tsne.fit_transform(X)

