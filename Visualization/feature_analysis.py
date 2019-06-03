# -*- coding: utf-8 -*-
import numpy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from grad_cam import GradCam

class FeatureAnalyzer:
    def __init__(self):
        pass

    def pca_analyze(self, x, n_dims=2):
        pca = PCA(n_components=n_dims)
        newX = pca.fit_transform(x)
        return x

    def tsne_analyze(self, x, n_dims=2):
        tsne = TSNE(n_components=n_dims, learning_rate=100)
        newX = tsne.fit_transform(x)
        return x

    def img_plot(self, path, x, y=None):
        if (y == None):
            plt.scatter(x[:, 0], x[:, 1], c="b")
            plt.show()
        else:
            plt.scatter(x[:, 0], x[:, 1], c=y)
            plt.colorbar()
            plt.show()
        plt.savefig(path)

    def grad_cam_analyze(self, path):
        gradCam = GradCam()
        gradCam.analyze(path)

"""
if __name__=="__main__":
    featureAnalyzer = FeatureAnalyzer()
    path = "cat_dog.png"
    featureAnalyzer.grad_cam_analyze(path)
"""
