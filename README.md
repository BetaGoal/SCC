# SCC
Predicting the Scalar Coupling Constants Between Atom Pairs in Molecules

Steps to run the code:
1. Download data from https://www.kaggle.com/c/champs-scalar-coupling/data and save them into './data/'.
2. Run visualization.py to merge and visualize the data.
3. Preprocess the data by command: 'python3 GP.py preprocess distance/coordinate'. Choose either distance or coordinate as argument.
4. Train the Gaussian Process Regression model by command: 'python3 GP.py train distance/coordinate'. Again, choose either distance or coordinate as argument. The trained model is saved as 'model_save.npy'.
5. Test the trained model by command: 'python3 GP.py test'.
