# Hybrid Artificial Neural Network Framework 

How to cite: "A Novel Hybrid Artificial Neural Network Framework for Classification Problems"

RL_MLP_SOM and RL_RBFN_SOM code files have been developed to make predictions on unbalanced and big data sets. In the developed integrated method, multilayer perceptron (MLP) and Radial basis function network (RBFN) algorithms were used for classification, while the feature selection was performed with the self-organizing map (SOM) integrated reinforcement learning (RL) approach. The steps of the Hybrid method are as follows:  

1. 20 percent of the data is randomly selected for the training set and the remaining 80 percent is allocated for the test set.
2. Half of the data allocated for training is used for training the network (RBFN or MLP) that will be used for prediction, and the other half is reserved for training the Kohonen network. The role of Kohonen network is to identify the “under-represented” class. The data allocated for the RBFN/MLP includes all the features, while the data that will be used to train Kohonen network involves only m number of features which is a subset of all of the features.
3. While forming the test set, m, which we specify as 10 here, random features are selected and a sub-test of the data set containing only the selected features is created.
4. A new sub-training set is created from the training set allocated to train the Kohonen network, including the same attributes in the sub-test data set, and the Kohonen network is trained with this new sub-training set. 
5. The sub-test of the data set is tested with this trained Kohonen network and as a result of the test, the samples that fall into the ”under represented set” are selected as ”potential under represented” data.
6. Using the IDs of the identified potential under represented entries, a data set containing all of the features is created.
7. Prediction is made for this data set using a trained feed-forward network (RBFN/MLP).
   - If the accuracy value is greater than a predetermined value, which we specify as 0.80 here and the Accuracy value of the under represented class is greater than a predetermined value, which we specify as 0.70 here, in the prediction results, feedback is given to the system with the reward mechanism. So in the next test set, the previously used selected feature set will be used.
   - If the accuracy value is not greater than the predetermined value and the accuracy value of the under represented class is not greater than its predetermined value in the estimation results, the selected feature set is changed randomly and steps 3 to 7 are repeated (For a test set, we have selected a maximum of 10 different random features).


Further detail is written as comment lines in the code.
 
