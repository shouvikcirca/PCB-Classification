# Imbalanced-Classification


Tried 3fold validation using Densenet 121
. High NonDefective Accuracy
. Low NonDefective Accuracy

Tried efficientnet B3 which was placed higher on Imagenet benchmark

Test set had to be balanced

Downsampled NonDefective class to balance the dataset and training on Densenet201.
1) removed random images directly from the dataset
2) randomly remove images in each epoch

Got results in 1) but found an issue. ImageDataGenerator by default downsizes images to 256X256. 
Will set to 600X600 and check.

Upsampled minority class by augmenting with horizontal flip.

Also Tried training an autoencoder for dimensionality reduction. Dataset not fitting in memory.

Note#
Data Augmentation is a technique to explicitly encode invariance into a ML model.
