# PCB-Classification

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


---Augmix-----
Further, networks trained on translation augmentations remain highly sensitive to images shifted by a single pixel
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations(Dan Hendrycks, Thomas Dietterich)


---SimCLR------
To evaluate the learned representations, we follow the widely used
linear evaluation protocol


Random Cropping might not be helpful as it might exclude the defect mark and that could lead to class inversion.


cmpth_checkpoints/exp2 was a mistake. Have to swap falseratio and trueratio.The ratio should be such that it magnifies
the minority-samples-class pred value and shrinks teh majority-samples-class pred value. In exp2 I did the opposite.



Looking at Defective and Non Defective Precision and Recall values in the results, one trens is emergent.
A lot of defective samples are being misclassiied as Non Defective. Owing to this, the defective precision is high
and defective precision is low. Because the number of false negatives is larger compared to number of false positives.

Due to the same reason, the result for non defective class is exactly opposite. Because in this case the number of false positives 
becomes larger than the number of false negatives.Thus Non Defective Precision values are lower and Non Defective Recall
values are higher.


Not a lot of difference when using Adam Optimizer and using training method described in densenet paper(without using mentioned initialization)


Softmax highlights the larger values while pushing the rest of the classes to very small values.
If we want to know how much the other classea are to each other, this thing has to be avoided.

If we make the values of the logits smaller before passing to softmax, the relative similarity can be retained.
This will make the distribution smoother.

Train and val accuraciea re going upto 95% but test acc is stuck at 89%.
Could be due to uncertainty introduced by labelling errors.


A larger variant of resnext gave an insignificant accuracy again. But the increase in number of parameters 
was significant. So there was no point training a larger model for the same performaance.


A larger vaaariant of resnext gave an insignificant accuracy again. But the increase in number of parameters 
was significant. So there was no point training a larger model for the same performaance.

Disadvantages of Batch Normalization
Paper: High Performance Large Scale Image Recognition Without Normalization

1) Incurs memory overhead
2) significantly increases the time required to calculate gradients in some networks
3) Introduces a discrepancy between the model during training and inference
4) Breaks the independence between training samples in the mini batch


Tried out Largin Margin Softmax Loss with different values for m (hyperparameter). 




