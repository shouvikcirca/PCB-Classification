# Imbalanced-Classification


Tried 3fold validation using Densenet 121
. High NonDefective Accuracy
. Low NonDefective Accuracy

Tried efficientnet B3 which was placed higher on Imagenet benchmark

Test set had to be balanced

Downsampled NonDefective class to balance the dataset
. removed random images directly from the dataset
. randomly remove images in each epoch

. Tried training an autoencoder for dimensionality reduction
