# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:14:32 2019

@author: yx199
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def t(filename):
    path = "./data/"
    train_part = pd.read_csv(path+filename)
    #print('Dim of '+filename+': ', train_part.shape)
    structures = pd.read_csv(path+'structures.csv')
    dipole_moments = pd.read_csv(path+'dipole_moments.csv')
    magnetic_shielding_tensors = pd.read_csv(path+'magnetic_shielding_tensors.csv')
    mulliken_charges = pd.read_csv(path+'mulliken_charges.csv')
    potential_energy = pd.read_csv(path+'potential_energy.csv')
    #print('Dim of structures: ', structures.shape)
    process = train_part.copy()
    merged_left = pd.merge(left=train_part, right=structures, how='left',
                           left_on=['molecule_name', 'atom_index_0'],
                           right_on=['molecule_name', 'atom_index'],
                           suffixes=(False, '_right'))
    merged_left = merged_left.drop(columns=['atom_index'])
    
    merged_all = pd.merge(left=merged_left, right=structures, how='left',
                          left_on=['molecule_name', 'atom_index_1'],
                          right_on=['molecule_name', 'atom_index'],
                          suffixes=('_1', '_2'))
    merged_all = merged_all.drop(columns=['atom_index'])
    
    merged_all = pd.merge(left=merged_all, right=potential_energy, how='left',
                          left_on=['molecule_name'],
                          right_on=['molecule_name'],
                          suffixes=(False, '_2'))
    
    merged_all = pd.merge(left=merged_all, right=dipole_moments, how='left',
                          left_on=['molecule_name'],
                          right_on=['molecule_name'],
                          suffixes=(False, '_dp'))
    
    merged_all = pd.merge(left=merged_all, right=mulliken_charges, how='left',
                          left_on=['molecule_name', 'atom_index_0'],
                          right_on=['molecule_name', 'atom_index'],
                          suffixes=(False, '_right'))
    merged_all = pd.merge(left=merged_all, right=mulliken_charges, how='left',
                          left_on=['molecule_name', 'atom_index_1'],
                          right_on=['molecule_name', 'atom_index'],
                          suffixes=('_1', '_2'))
    
    merged_all = merged_all.drop(columns=['atom_index_0','atom_index_1','atom_index_2'])                      
    merged_all = merged_all.drop(columns = ['id'])
    merged_all['distance'] = np.sqrt((merged_all['x_1']-merged_all['x_2'])**2 +\
              (merged_all['y_1']-merged_all['y_2'])**2 +\
              (merged_all['z_1']-merged_all['z_2'])**2)
    merged_all['mlken_diff'] = merged_all['mulliken_charge_1'] - merged_all['mulliken_charge_2']
    
    print(merged_all.columns)
    print('---------------------------------------------------------------------')
    print('Dimension after merging: ', merged_all.shape)
    
    merged_all.to_csv('merged_all.csv')  
    
    
def prt(merge_all):
    print("\nDimension: ", merge_all.shape)
    print("\n")
    print("========================== Overview ==================================\n")
    print(merge_all.head())
    print("\n===================== Type of each Variable =====================\n")
    print(merge_all.info(null_counts = True))
    print("\n======================= Missing Data =============================")
    print(merge_all.isnull().sum())
    print("\n======================== Summary ===========================\n")
    print("***** Numerical Variables *****\n")
    print(pd.concat([merge_all.describe(), pd.DataFrame(
    merge_all.mad(),columns = ["Mad"]).T,
    pd.DataFrame(merge_all.kurt(),columns = ["Kurtosis"]).T,
    pd.DataFrame(merge_all.skew(),columns = ["Skew"]).T]))
    print("\n***** Categorical Variables *****\n")
    print(merge_all.describe(include = ['O']))
    print('\n')

def plot(merge_all, numerical_feats, categorical_feats):    
    merge_all[numerical_feats].hist(figsize=(30, 20), bins = 50)
    # plt.savefig('hist_num.pdf')
    plt.show()
    
    fig, ax = plt.subplots(1, len(categorical_feats))
    for i, categorical_feature in enumerate(merge_all[categorical_feats]):
        merge_all[categorical_feature].value_counts().plot(
                "bar", ax=ax[i], figsize=(20, 10)).set_title(categorical_feature)
    # fig.savefig('hist_cat.pdf')
    fig.show()
    # ############################# Matrix Plot ##################################
    pt = pd.plotting.scatter_matrix(merge_all, alpha=0.2, figsize=(15, 15))
    # plt.savefig('Matrix_Plot.pdf')
    plt.show()
    # ######################### Correlation Plot ################################
    colormap = plt.cm.RdBu
    plt.figure(figsize=(32,10))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(merge_all[numerical_feats].corr(),linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    # plt.savefig('Correlation_Plot.pdf')
    plt.show()
    
def feats(df):
    numerical_feats = df._get_numeric_data().columns.tolist()
    categorical_feats = df.select_dtypes(include=['object']).columns.tolist()
    categorical_feats.remove('molecule_name')
    return numerical_feats,categorical_feats

t('train.csv')
train = pd.read_csv('merged_all.csv')
train = train.drop(train.columns[[0]],1)
(nf,catf) = feats(train)
print(catf)
# visualization
plot(train,nf,catf)
