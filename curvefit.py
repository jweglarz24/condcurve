#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:28:11 2025

@author: jweglarz
"""

#%% Boot

#import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import numpy as np
import h5py

file_path = '../output/FineRes_1/CondCurveDB_FineRes1.hdf5'
overwrite = True #if True, overwrites current curve-fit database info

bounds = [0.8, 1.2]

#%% Populate info from .hdf5

def populate(file_path):
    
    c_to_os = []
    metallicities = []
    list_of_condensates = []
    
    # Populate list of C/O, Fe/H, condensates
    with h5py.File(file_path, 'a') as f:
        #Populate C/O
        for CO in f:
            #CO = key.split('_')[1]
            c_to_os.append(CO)
        #Populate Fe/H
        first_CO = next(iter(f))
        for FEH in f[first_CO]:
            #FEH = key.split('_')[1]
            metallicities.append(FEH)
        #Populate conds
        first_FEH = next(iter(f[first_CO]))
        for cond in f[first_CO][first_FEH]:
            list_of_condensates.append(cond)
            
    return c_to_os, metallicities, list_of_condensates

c_to_os, metallicities, list_of_condensates = populate(file_path)
#print(f'{file_path} opened successfully. Performing curve-fit...')

#%% Curve-fit functions
    
def fit_h1(T, P, F, C):
    Y = 1e4 / np.array(T)
    
    # Convert to arrays
    F = np.array(F)
    C = np.array(C)
    logP = np.log(P)
    logP2 = logP**2
    
    # Build matrix
    X = np.column_stack([
        # a0 terms
        np.ones(len(Y)),
        F,
        C,
        # a1 terms
        -logP,
        -logP * F,
        -logP * C,
        # a2 terms
        -logP2,
        -logP2 * F,
        -logP2 * C
    ])
    
    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    # Reshape into 3x4 array: [a0_coeffs, a1_coeffs, a2_coeffs]
    coeffs_structured = coeffs.reshape(3, 3)
    
    return coeffs_structured, residuals

def fit_h2(T, P, F, C):
    Y = 1e4 / np.array(T)
    
    # Convert to arrays
    F = np.array(F)
    C = np.array(C)
    logP = np.log(P)
    logP2 = logP**2
    
    # Build matrix
    X = np.column_stack([
        # a0 terms
        np.ones(len(Y)),
        F,
        C,
        F*C,
        # a1 terms
        -logP,
        -logP * F,
        -logP * C,
        -logP * F * C,
        # a2 terms
        -logP2,
        -logP2 * F,
        -logP2 * C,
        -logP2 * F * C
    ])
    
    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    # Reshape into 3x4 array: [a0_coeffs, a1_coeffs, a2_coeffs]
    coeffs_structured = coeffs.reshape(3, 4)
    
    return coeffs_structured, residuals

def fit_func_h(coeffs, P, F, C):
    """
    coeffs : array, shape (3, 4)
        Coefficient matrix where:
        coeffs[0,:] = [b00, b01, b02, b03] for a0
        coeffs[1,:] = [b10, b11, b12, b13] for a1
        coeffs[2,:] = [b20, b21, b22, b23] for a2
    """
    
    F = np.array(F)
    C = np.array(C)
    logP = np.log(P)
    
    if np.shape(coeffs) == (3,3):
        # Calculate a0, a1, a2 using Shape 1
        a0 = coeffs[0, 0] + coeffs[0, 1]*F + coeffs[0, 2]*C
        a1 = coeffs[1, 0] + coeffs[1, 1]*F + coeffs[1, 2]*C
        a2 = coeffs[2, 0] + coeffs[2, 1]*F + coeffs[2, 2]*C
    elif np.shape(coeffs) == (3,4):
        a0 = coeffs[0, 0] + coeffs[0, 1]*F + coeffs[0, 2]*C + coeffs[0,3]*F*C
        a1 = coeffs[1, 0] + coeffs[1, 1]*F + coeffs[1, 2]*C + coeffs[1,3]*F*C
        a2 = coeffs[2, 0] + coeffs[2, 1]*F + coeffs[2, 2]*C + coeffs[2,3]*F*C
    else:
        print('Coefficient matrix = ')
        print(coeffs)
        raise ValueError('Coefficient matrix of unknown shape.')
        
    # Calculate temperature
    T = 1e4 / (a0 - a1*logP - a2*logP**2)
    
    return T

#%% Curve-fit function

def fit_and_store(list_of_condensates, c_to_os, metallicities):
    refit_conds = []
    
    # Open .hdf5
    with h5py.File(file_path, 'a') as f:
        # For every condensate, extract primary Temp/Pres/CO/FEHs for curve fitting
        for i, cond in enumerate(list_of_condensates):
            flag_Refit = 0
            T1, P1, F1, C1 = [], [], [], []
            for CO in c_to_os:
                for FEH in metallicities:
                    # Extract data
                    data = f[CO][FEH][cond]
                    isPrimary = data['isPrimary'][()]
                    
                    if len(data['T'][()][isPrimary])>1:
                        T1.extend(data['T'][()][isPrimary])
                        P1.extend(data['P'][()][isPrimary])
                        F1.extend( [np.log10(float(FEH.split('_')[1]))] * len(data['T'][()][isPrimary]) )
                        C1.extend( [float(CO.split('_')[1])]  * len(data['T'][()][isPrimary]) )
            
            # Create linear fit using extracted data
            # reslim = 0.9
            
            if len(T1) > 9:
                coeffs, residuals = fit_h2(T1, P1, F1, C1)
                
                # if residuals/len(T1) > reslim:
                #     coeffs, residuals = fit_h2(T1, P1, F1, C1)
                    
                #     if residuals/len(T1) > reslim:
                #         print(f'{cond} fit not satisying.')
                #         print(residuals/len(T1))
                        
            else:
                coeffs = np.full((3, 4), np.nan)
            
            print(f'\tCurve-fit for condensate {i+1}/{len(list_of_condensates)} completed. Storing fit parameters...')
            
            # Store linfit data within .hdf5 for every cond instance
            for CO in c_to_os:
                for FEH in metallicities:
                    # Extract data from .hdf5             
                    data = f[CO][FEH][cond]
                    isPrimary = data['isPrimary'][()]
                    
                    temp = np.where(isPrimary, data['T'][()], np.nan) #Primary temp only, rest np.nan
                    pres = np.where(isPrimary, data['P'][()], np.nan) #Primary pres only, rest np.nan
                    metallicity = np.log10(float(FEH.split('_')[1])) #Converting x-solar to dex
                    c_to_o = float(CO.split('_')[1])
                    
                    # Calculate fit temperatures using coefficients
                    T_fit = fit_func_h(coeffs, pres, metallicity, c_to_o)
                    
                    # Create/overwrite fit datasets within .hdf5
                    if overwrite:
                        if 'fitTemp' in data:
                            del(data['fitTemp'])
                        if 'fitCoeffs' in data:
                            del(data['fitCoeffs'])
                        if 'fitCoeffs1' in data:
                            del(data['fitCoeffs1'])
                        if 'fitCoeffs2' in data:
                            del(data['fitCoeffs2'])
                        if 'fitCoeffs3' in data:
                            del(data['fitCoeffs3'])
                        if 'fitR2' in data:
                            del(data['fitR2'])
                        if 'regime' in data:
                            del(data['regime'])
                            
                    fitTemp = data.require_dataset('fitTemp', 
                                       shape = np.shape(temp),
                                       dtype = float)
                                        
                    fitCoeffs = data.require_dataset('fitCoeffs',
                                                     shape = np.shape(coeffs),
                                                     dtype = float)
                    fitCoeffs[:] = coeffs # Store coefficients regardless of curve presence
                    
                    fitR2 = data.require_dataset('fitR2',
                                                 shape=(1,),
                                                 dtype = float)
                    
                    # If curve exists, store fit data & stat functions
                    mask = ~np.isnan(temp) & ~np.isnan(T_fit)
                    if len(temp[mask])>1:
                        R2 = r2_score(temp[mask], T_fit[mask])
                        
                        if R2 < 0.8:
                            if flag_Refit == 0:
                                refit_conds.append(cond)
                                print(f'{cond.strip()} in need of refitting.')
                                flag_Refit += 1
                        
                        fitTemp[:] = T_fit
                        fitR2[:] = [float(R2)]
                        
            if flag_Refit == 0 and len(T1)>9:
                print(f'{cond.strip()} exists and fits well.')
                
    return refit_conds


def fit_and_store_piecewise(list_of_condensates, c_to_os, metallicities, bounds):
    refit_conds = []
    
    # Open .hdf5
    with h5py.File(file_path, 'a') as f:
        # For every condensate, extract primary Temp/Pres/CO/FEHs for curve fitting
        for i, cond in enumerate(list_of_condensates):
            flag_Refit = 0
            
            T1, P1, F1, C1 = [], [], [], []
            T2, P2, F2, C2 = [], [], [], []
            T3, P3, F3, C3 = [], [], [], []
            for CO in c_to_os:
                for FEH in metallicities:
                    # Extract data
                    data = f[CO][FEH][cond]
                    isPrimary = data['isPrimary'][()]
                    
                    co = float(CO.split('_')[1])
                    if co < bounds[0]:
                        if len(data['T'][()][isPrimary])>1:
                            T = data['T'][()]
                            
                            T1.extend(data['T'][()][~np.isnan(T)])
                            P1.extend(data['P'][()][~np.isnan(T)])
                            F1.extend( [np.log10(float(FEH.split('_')[1]))] * len(data['T'][()][~np.isnan(T)]) )
                            C1.extend( [float(CO.split('_')[1])]  * len(data['T'][()][~np.isnan(T)]) )
                    elif (co >= bounds[0] and co <= bounds[1]):
                        if len(data['T'][()][isPrimary])>1:
                            T = data['T'][()]
                            
                            T2.extend(data['T'][()][~np.isnan(T)])
                            P2.extend(data['P'][()][~np.isnan(T)])
                            F2.extend( [np.log10(float(FEH.split('_')[1]))] * len(data['T'][()][~np.isnan(T)]) )
                            C2.extend( [float(CO.split('_')[1])]  * len(data['T'][()][~np.isnan(T)]) )
                    elif co > bounds[1]:
                        if len(data['T'][()][isPrimary])>1:
                            T = data['T'][()]
                            
                            T3.extend(data['T'][()][~np.isnan(T)])
                            P3.extend(data['P'][()][~np.isnan(T)])
                            F3.extend( [np.log10(float(FEH.split('_')[1]))] * len(data['T'][()][~np.isnan(T)]) )
                            C3.extend( [float(CO.split('_')[1])]  * len(data['T'][()][~np.isnan(T)]) )
                
            if len(T1) > 9:
                coeffs1, res1 = fit_h2(T1, P1, F1, C1)
            else:
                coeffs1 = np.full((3, 3), np.nan)
                
            if len(T2) > 9:
                coeffs2, res2 = fit_h2(T2, P2, F2, C2)
            else:
                coeffs2 = np.full((3, 3), np.nan)
                
            if len(T3) > 9:
                coeffs3, res3 = fit_h2(T3, P3, F3, C3)
            else:
                coeffs3 = np.full((3, 3), np.nan)
                
            print(f'\tCurve-fit for condensate {i+1}/{len(list_of_condensates)} completed. Storing fit parameters...')
            
            # Store linfit data within .hdf5 for every cond instance
            for CO in c_to_os:
                for FEH in metallicities:
                    # Extract data from .hdf5             
                    data = f[CO][FEH][cond]
                    isPrimary = data['isPrimary'][()]
                    
                    temp = np.where(isPrimary, data['T'][()], np.nan) #Primary temp only, rest np.nan
                    pres = np.where(isPrimary, data['P'][()], np.nan) #Primary pres only, rest np.nan
                    metallicity = np.log10(float(FEH.split('_')[1])) #Converting x-solar to dex
                    c_to_o = float(CO.split('_')[1])
                    
                    # Calculate fit temperatures using coefficients
                    if c_to_o < bounds[0]:
                        T_fit = fit_func_h(coeffs1, pres, metallicity, c_to_o)
                    elif (c_to_o >= bounds[0] and c_to_o <= bounds[1]):
                        T_fit = fit_func_h(coeffs2, pres, metallicity, c_to_o)
                    elif c_to_o > bounds[1]:
                        T_fit = fit_func_h(coeffs3, pres, metallicity, c_to_o)
                    
                    # Create/overwrite fit datasets within .hdf5
                    if overwrite:
                        if 'fitTemp' in data:
                            del(data['fitTemp'])
                        if 'fitCoeffs' in data:
                            del(data['fitCoeffs'])
                        if 'fitCoeffs1' in data:
                            del(data['fitCoeffs1'])
                        if 'fitCoeffs2' in data:
                            del(data['fitCoeffs2'])
                        if 'fitCoeffs3' in data:
                            del(data['fitCoeffs3'])
                        if 'fitR2' in data:
                            del(data['fitR2'])
                        if 'regime' in data:
                            del(data['regime'])
                            
                    fitTemp = data.require_dataset('fitTemp', 
                                       shape = np.shape(temp),
                                       dtype = float)
                    
                    if c_to_o < bounds[0]:
                        data.attrs['regime'] = '1'
                            
                    elif (c_to_o >= bounds[0] and c_to_o <= bounds[1]):
                        data.attrs['regime'] = '2'
                        
                    elif c_to_o > bounds[1]:
                        data.attrs['regime'] = '3'
                        
                    fitCoeffs1 = data.require_dataset('fitCoeffs1',
                                                     shape = np.shape(coeffs1),
                                                     dtype = float)
                    fitCoeffs1[:] = coeffs1 # Store coefficients regardless of curve presence
                    
                    fitCoeffs2 = data.require_dataset('fitCoeffs2',
                                                     shape = np.shape(coeffs2),
                                                     dtype = float)
                    fitCoeffs2[:] = coeffs2 # Store coefficients regardless of curve presence
                        
                    fitCoeffs3 = data.require_dataset('fitCoeffs3',
                                                     shape = np.shape(coeffs3),
                                                     dtype = float)
                    fitCoeffs3[:] = coeffs3 # Store coefficients regardless of curve presence
                    
                    fitR2 = data.require_dataset('fitR2',
                                                 shape=(1,),
                                                 dtype = float)
                    
                    # If curve exists, store fit data & stat functions
                    mask = ~np.isnan(temp) & ~np.isnan(T_fit)
                    if len(temp[mask])>1:
                        R2 = r2_score(temp[mask], T_fit[mask])
                        
                        if R2 < 0.7:
                            if flag_Refit == 0:
                                refit_conds.append(cond)
                                print(f'{cond.strip()} in need of refitting.')
                                flag_Refit += 1
                        
                        fitTemp[:] = T_fit
                        fitR2[:] = [float(R2)]
                        
            if flag_Refit == 0 and (len(T1)>9 or len(T2)>9 or len(T3)>9):
                print(f'{cond.strip()} exists and fits well.')
                
    return refit_conds
            

#%% Curve-fitting

refit_conds_1 = fit_and_store(list_of_condensates, c_to_os, metallicities)
#print(f'Refitting condensates: {refit_conds_1}')
#refit_conds_2 = fit_and_store_piecewise(refit_conds_1, c_to_os, metallicities, bounds=bounds)
#print(f'Condensates in need of manual refitting: {refit_conds_2}')          