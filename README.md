# Siamese CNN for animal identification

The siamese approach allows to learn a meaningful embedding by training a CNN with pairs of genuine and impostor images. Weights are shared during training so that pairs of images will be represented according to both their features and the label that identifies the pair as of two images representing the same object or two different ones.

For details, refer to [Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to face verification." Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005.](http://ftp.cs.nyu.edu/~sumit/publications/assets/cvpr05.pdf)

## So, what?

Here are collected a series of scripts in which the contrastive loss approach
has been explored while developing the algorithm for [idtracker.ai](www.idtracker.ai).

The code, written for tensorflow0.6 uses only basic functions and it is heavily commented. It includes a study on the contrastive loss and its parameters.

## Understanting contrastive loss

The function plotter allows to test and plot in a 3d scatter the contrastive loss
implemented in
```
plotter.contrastive_loss1
```

## Training

In this implementation we used dataset of images generated in matlab, however
it should be easy to rearrange the code in order to load images or binary files.

## Classification

Test images can be classified by using KNN or hierarchical clustering
in siamese_model_HD_KNN and siamese_model_HD_Clustering, respectively. Caveat:
The two scripts are as self-contained as reduntant.
