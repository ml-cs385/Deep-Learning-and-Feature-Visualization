## This is an API for feature analysis, including PCA, TSNE and GradCam

Usage is simple, you can call the three methods by:

featureAnalyzer = FeatureAnalyzer()
x = ... # The latent vector you obtained
y = ... # The corresponding category for the x, if not obtained, set it to be None
n = 2   # The number of dimensions you want to reduce the vector to
savePath1, savePath2 = ..., ... # The paths you want to save the figure

# Call PCA
newPCAX = featureAnalyzer.pca_analyze(x, n_dims=n)
featureAnalyzer.img_plot(savePath1, newPCAX, y)

# Call TSNE
newTSNEX = featureAnalyzer.tsne_analyze(x, n_dims=n)
featureAnalyzer.img_plot(savePath2, newTSNEX, y)

# Call GradCam
path = ...
featureAnalyzer.grad_cam_analyze(path)


