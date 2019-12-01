# SCC
Predicting the Scalar Coupling Constants Between Atom Pairs in Molecules

Steps to run the code:
1. Run visualization.py to merge the data.
2. Preprocess the data by command: python3 GP.py preprocess distance/coordinate. Choose either distance or coordinate as argument.
3. Train the Gaussian Process Regression model by command: python3 GP.py train distance/coordinate. Again, choose either distance or coordinate as argument. The trained model is saved as model_save.npy.
4. Test the trained model by command: python3 GP.py test.
