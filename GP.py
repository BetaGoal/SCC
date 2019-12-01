# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:15:18 2019

@author: yx199
"""
import numpy as np
import pandas as pd
from preprocessing import encode, outliers
import GPy
from sklearn.metrics import mean_squared_error
from sys import argv
import os

def preprocess(inputfile, select, frac = 0.01):
    merged_all = pd.read_csv(inputfile)
    merged_all = merged_all.drop(merged_all.columns[[0]],1)
    # sample 1% due to memory limitation
    merged_all = merged_all.sample(frac=frac, replace=False, random_state=1234)
    atoms = merged_all.groupby('atom_2')
    C = atoms.get_group('C')
    N = atoms.get_group('N')
    H = atoms.get_group('H')
    print(C.shape)
    print(N.shape)
    print(H.shape)
    merged_smp_C_test = C.sample(frac=0.2, replace=False, random_state=1234)
    merged_smp_N_test = N.sample(frac=0.2, replace=False, random_state=1234)
    merged_smp_H_test = H.sample(frac=0.2, replace=False, random_state=1234)
    merged_smp_C_train = C.drop(merged_smp_C_test.index)
    merged_smp_N_train = N.drop(merged_smp_N_test.index)
    merged_smp_H_train = H.drop(merged_smp_H_test.index)
    merged_smp_test = pd.concat([merged_smp_C_test, merged_smp_N_test, merged_smp_H_test])
    merged_smp_train = pd.concat([merged_smp_C_train, merged_smp_N_train, merged_smp_H_train])
    
    merged_smp_test = merged_smp_test.drop(['potential_energy','atom_1', 'atom_2','molecule_name','X','Y','Z'], axis=1)
    merged_smp_train = merged_smp_train.drop(['potential_energy','atom_1', 'atom_2','molecule_name','X','Y','Z'], axis=1)
    
    print('\n======================== test:===========================\n')
    test = encode(merged_smp_test)
    print('\n======================== train:===========================\n')
    train = encode(merged_smp_train)
    if select == 'distance':
        test = test.drop(['x_1','x_2','y_1','y_2','z_1','z_2'], axis=1)
        train = train.drop(['x_1','x_2','y_1','y_2','z_1','z_2'], axis=1)
    elif select == 'coordinate':
        test = test.drop(['distance'], axis=1)
        train = train.drop(['distance'], axis=1)
    else:
        print('Select distance or coordinate as input')
    
    #"""
    print('\n======================== test:===========================\n')
    test = outliers(test,'test')
    print('\n======================== train:===========================\n')
    train = outliers(train,'train')
    #"""
    
# create GP model
def model(X,Y,K):
    m = GPy.models.GPRegression(X,Y,K)
    m.optimize(messages=True)
    # 1: Saving a model:
    np.save('model_save.npy', m.param_array)
    return m

    
def test(model, X_train, y_train, X_test, y_test):
    m_load = GPy.models.GPRegression(X_train, y_train, initialize=False)
    m_load.update_model(False) # do not call the underlying expensive algebra on load
    m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
    m_load[:] = np.load(model) # Load the parameters
    m_load.update_model(True) # Call the algebra only once
    print(m_load)
    
    y_pred, sigms = m_load.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: ', mse)
    print('mean: ', y_test.mean())

if __name__ == '__main__':  
    if argv[1] == 'preprocess':
        file = 'merged_all.csv'
        preprocess(file, argv[2])  
    if os.path.isfile('train.csv'):        
        test_data = pd.read_csv('test.csv')
        train_data = pd.read_csv('train.csv')        
        y_test = np.transpose(np.array([test_data.scalar_coupling_constant]))
        y_train = np.transpose(np.array([train_data.scalar_coupling_constant]))
        X_train = train_data.drop(columns = ['id', 'scalar_coupling_constant','mlken_diff'])
        X_train = np.array(X_train.drop(X_train.columns[[0]],1))
        X_test = test_data.drop(columns = ['id', 'scalar_coupling_constant','mlken_diff'])
        X_test = np.array(X_test.drop(X_test.columns[[0]],1))
    else:
        print('Preprocess data using "GP preprocess"')
    if argv[1] == 'train':
        if argv[2] == 'distance':
            dim = 11
        else:
            dim = 16
        m = model(X_train,y_train,GPy.kern.RBF(dim))
    if argv[1] == 'test':
        if os.path.isfile('model_save.npy'):  
            test('model_save.npy',X_train, y_train, X_test, y_test)
        else:
            print('Trian model using "GP train"')
    
    
    

    
    
    
    