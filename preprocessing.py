# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:31:40 2019

@author: yx199
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def encode(merged_smp):
    print('\nDimension after sample: ', merged_smp.shape)
    print("\n======================== Summary ===========================\n")
    print("***** Numerical Variables *****\n")
    print(pd.concat([merged_smp.describe(), 
                     pd.DataFrame(merged_smp.mad(),columns = ["Mad"]).T,
                     pd.DataFrame(merged_smp.kurt(),columns = ["Kurtosis"]).T,
                     pd.DataFrame(merged_smp.skew(),columns = ["Skew"]).T]))
    print("\n***** Categorical Variables *****\n")
    print(merged_smp.describe(include = ['O']))
    ids = list(range(1,merged_smp.shape[0]+1))
    ## One-Hot Encoding of categorical variables
    merged_en = pd.get_dummies(merged_smp, drop_first = False)
    # Scale numerical data
    merged_sc = merged_en.copy()
    merged_sc.x_1 = StandardScaler().fit_transform(merged_sc.x_1.values.reshape(-1,1))
    merged_sc.x_2 = StandardScaler().fit_transform(merged_sc.x_2.values.reshape(-1,1))
    merged_sc.y_1 = StandardScaler().fit_transform(merged_sc.y_1.values.reshape(-1,1))
    merged_sc.y_2 = StandardScaler().fit_transform(merged_sc.y_2.values.reshape(-1,1))
    merged_sc.z_1 = StandardScaler().fit_transform(merged_sc.z_1.values.reshape(-1,1))
    merged_sc.z_2 = StandardScaler().fit_transform(merged_sc.z_2.values.reshape(-1,1))
    merged_sc.distance = StandardScaler().fit_transform(merged_sc.distance.values.reshape(-1,1))
    merged_sc.mlken_diff = StandardScaler().fit_transform(merged_sc.mlken_diff.values.reshape(-1,1))
    merged_sc.mulliken_charge_1 = StandardScaler().fit_transform(merged_sc.mulliken_charge_1.values.reshape(-1,1))
    merged_sc.mulliken_charge_1 = StandardScaler().fit_transform(merged_sc.mulliken_charge_1.values.reshape(-1,1))
    merged_sc.scalar_coupling_constant = StandardScaler().fit_transform(merged_sc.scalar_coupling_constant.values.reshape(-1,1))
    col = ['mulliken_charge_1','mulliken_charge_2','distance','mlken_diff','scalar_coupling_constant',
           'x_1','x_2','y_1','y_2','z_1','z_2']
    for x in col:
        merged_sc[x] = MinMaxScaler().fit_transform(merged_sc[x].values.reshape(-1,1))
    print('\n======================== After encoding ======================== \n')
    print('Dimension after Encoding: ', merged_sc.shape)
    print(merged_sc.columns)
    print("\n======================== Summary ===========================\n")
    print(merged_sc.describe())
    return merged_sc

def outliers(sc,w):
    merged_full = sc.copy()
    id_full = list(range(1,merged_full.shape[0]+1))
    lab_full = merged_full.scalar_coupling_constant
    var_full = merged_full.drop(['scalar_coupling_constant'], axis=1)
    #################### Isolation Forest #########################
    var_IF = var_full.copy()
    clf_if = IsolationForest(n_estimators=500, max_samples=1.0,contamination=0.01, bootstrap=True, n_jobs = -1)
    clf_if.fit(var_IF.values, lab_full)
    score = clf_if.decision_function(var_IF.values) # Isolation Forest score
    original_paper_score = [0.5-s for s in score]
    isof = clf_if.predict(var_IF.values) # Return -1 for outliers and 1 for inliers
    var_IF.insert(0, 'scalar_coupling_constant', lab_full)
    var_IF.insert(0, 'id', id_full)
    var_IF['isof'] = isof
    myindex = var_IF['isof'] > 0 # 1 is True (inlier) & -1 is False (outlier)
    out_IF = var_IF[isof<0]
    out_IF.drop('isof', axis=1, inplace=True)
    train_IF = var_IF.loc[myindex]
    train_IF.reset_index(drop=True, inplace=True)
    train_IF.drop('isof', axis=1, inplace=True)
    print('Isolation Forest - Numbers of outliers found: %d / %d (%d)' % (out_IF.shape[0], train_IF.shape[0], var_IF.shape[0]))
    print('-----------------------------------------------')
    #################### Local Outlier Factor #########################
    var_lof = var_full.copy()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, n_jobs = -1)
    y_pred = lof.fit_predict(var_lof.values, lab_full)
    ### Returns -1 for anomalies/outliers and 1 for inliers
    var_lof.insert(0, 'scalar_coupling_constant', lab_full)
    var_lof.insert(0, 'id', id_full)
    train_lof = var_lof[y_pred == 1]
    out_lof = var_lof[y_pred == -1]
    print('Local Outlier Factor - Numbers of outliers found: %d / %d (%d)' % (out_lof.shape[0], train_lof.shape[0], var_lof.shape[0]))
    print('-----------------------------------------------')
    # Interction of two methods
    IF_lof = set(out_IF.id).intersection(out_lof.id)
    print('Intersection outlier: ',len(IF_lof))
    merged_out = pd.concat([train_IF,train_lof],ignore_index=True).drop_duplicates().reset_index(drop=True)
    print('Dimension after removing outliers: ', merged_out.shape)
    print(merged_out.columns)
    merged_out.to_csv(w + '.csv')
    return merged_out

