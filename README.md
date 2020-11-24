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

With 1794 samples in training set, takes about 1600 seconds per epoch


Decided to try cutmix augmentation.

Note
The problem with having an imbalanced dataset in a binary classification task is that
it is not that the model is learning the discriminative features of the majority class well.
Actually, it is badly learning the discriminative features of the minority class.
Because we have only two classes, failure to identify a sample as belonging to one class(minority)
automatically classifies it to be a sample of the other class(majority class). So there is a very heavy bias
towards the majority class.
