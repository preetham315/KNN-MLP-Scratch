## KNN

### Approach

I initially started with examining the data for better understanding of the train and test data that we are using then I moved to the fit where in we are assigning the the train of data and target. Then in predict I used the manhattan distance and euclidean distance to calculate the nearest k neighbors. so in predict i initially created a empty list for the predicted y. then initialised one varialble and two lists. SO here I am doing calculcating od distances and parlley appending that to the distance_target list and then sorting based on the list 1 element and after doing that I am getting only the k nearest neighbors using the for loop after this i am getting the most frequent element and appending it to the ypred list. this will be the final list with predicted values.


So for this problem i initially used the np.sum and np.square and the rest of the np functions for calculating the eucleadian and manhattan distances so it was taking a 40 misnutes to give the output later i changed this to normal math functions which helped in reducing the time taken to run the program to 5 and then finally after changes again it came down to 2to 3 minutes.

### Final Result

Obtained an accuracy of 100 for iris data set and 97 % for digits


## Multi Layer Perceptron

### Appraoch

So after a lot of reading and understanding I started with the activation functions for this program. I worte the activation functions for normal call and when the derivative is true. Then i moved to fir where in i started with initialising the weights of of the input layer and hidden layer. After that I came back the the fir and started coding the function for fit where in we will be considering the matrix created in initialise. Based on the activation function (hidden and output) we will be considering the respective functions and calculating the forward propogation once this is done then we will calculate the cross entropy loss and the gradient of this which is used in calculating the sigmoid function

I faced lot of issues while calculating the back propagation function.

Then in predict we just simply write the feed forward


### References

* https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
* https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2 (KNN- code ref)
* https://blog.devgenius.io/implementing-k-nearest-neighbors-from-scratch-in-python-d5eaaf558d49 (KNN- code ref)
* https://www.digitalocean.com/community/tutorials/k-nearest-neighbors-knn-in-python
* https://www.analyticsvidhya.com/blog/2021/01/a-quick-introduction-to-k-nearest-neighbor-knn-classification-using-python/
* https://www.ritchieng.com/machine-learning-k-nearest-neighbors-knn/
* https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python
* https://towardsdatascience.com/secret-sauce-behind-the-beauty-of-deep-learning-beginners-guide-to-activation-functions-a8e23a57d046#:~:text=Identity%20or%20Linear%20Activation%20Function,proportional%20to%20the%20input%20data.
* https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
* https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch (Code- ref_mlp)
* https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
* https://www.statology.org/one-hot-encoding-in-python/
* https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/supervised_learning/multilayer_perceptron.py#L27 (code- ref-mlp)
* https://chat.openai.com/chat (reference for mlp resources)
* https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd
* https://neuralthreads.medium.com/backpropagation-made-super-easy-for-you-part-2-7b2a06f25f3c
* https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9