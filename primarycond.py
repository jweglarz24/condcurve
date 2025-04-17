#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:04:43 2025

@author: jweglarz
"""

# Default
import pyfastchem
import numpy as np
#import matplotlib.pyplot as plt
#import csv
import scipy.io
import os
import glob
import re
import copy

# Double-dictionary
from dataclasses import dataclass, field
from collections import defaultdict

list_of_condensates = ['Al(s)', 'AlClO(s)', 'AlCl3(s,l)', 'KAlCl4(s)', 'NaAlCl4(s)', 'K3AlCl6(s)', 'Na3AlCl6(s)', 'AlF3(s,l)', 'K3AlF6(s)', 'Na3AlF6(s,l)', 'AlN(s)', 'NaAlO2(s)', 'K3Al2Cl9(s)', 'MgAl2O4(s,l)', 'Al2O3(s,l)', 'Al2SiO5(s)', 'Al2S3(s)', 'Na5Al3F14(s,l)', 'Al6Si2O13(s)', 'C(s)', 'CuCN(s)', 'KCN(s,l)', 'K2CO3(s,l)', 'MgCO3(s)', 'NaCN(s,l)', 'NaCO3(s,l)', 'SiC(s)', 'TiC(s,l)', 'Cr3C2(s)', 'MgC2(s)', 'Al4C3(s)', 'Cr7C3(s)', 'Mg2C3(s)', 'Ni(CO)4(l)', 'Fe(CO)5(l)', 'Cr23C6(s)', 'Ca(s,l)', 'CaCl2(s,l)', 'CaF2(s,l)', 'Ca(OH)2(s)', 'CaO(s,l)', 'CaS(s)', 'CuCl(s,l)', 'NH4Cl(s)', 'NH4ClO4(s)', 'KCl(s,l)', 'KClO4(s)', 'NaCl(s,l)', 'NaClO4(s)', 'CoCl2(s,l)', 'CuCl2(s)', 'FeCl2(s,l)', 'MgCl2(s,l)', 'NiCl2(s,l)', 'SCl2(l)', 'ClSSCl(l)', 'TiCl2(s)', 'FeCl3(s,l)', 'TiCl3(s)', 'TiCl4(s,l)', 'Co(s,l)', 'CoF2(s,l)', 'CoF3(s)', 'CoO(s)', 'CoSO4(s)', 'Co3O4(s)', 'Cr(s,l)', 'CrN(s)', 'Cr2N(s)', 'Cr2O3(s,l)', 'Cu(s,l)', 'CuF(s)', 'CuF2(s,l)', 'Cu(OH)2(s)', 'CuO(s)', 'CuSO4(s)', 'Cu2O(s,l)', 'KF(s,l)', 'NaF(s,l)', 'FeF2(s,l)', 'K(HF2)(s,l)', 'MgF2(s,l)', 'FeF3(s)', 'TiF3(s)', 'TiF4(s)', 'Fe(s,l)', 'Fe(OH)2(s)', 'Fe(OH)3(s)', 'FeO(s,l)', 'FeSO4(s)', 'FeS(s,l)', 'FeS2(s)', 'Fe2O3(s)', 'Fe2(SO4)3(s)', 'Fe3O4(s)', 'KH(s)', 'KOH(s,l)', 'NaH(s)', 'NaOH(s,l)', 'MgH2(s)', 'Mg(OH)2(s)', 'O2S(OH)2(s,l)', 'TiH2(s)', 'H3PO4(s,l)', 'N2H4(l)', 'H2SO4.H2O(s,l)', 'H2SO4.2H2O(s,l)', 'H2SO4.3H2O(s,l)', 'H2SO4.4H2O(s,l)', 'K(s,l)', 'KO2(s)', 'K2O(s)', 'K2O2(s)', 'K2SiO3(s,l)', 'K2SO4(s,l)', 'K2S(s,l)', 'Mg(s,l)', 'MgO(s,l)', 'MgSiO3(s,l)', 'MgTiO3(s,l)', 'MgSO4(s,l)', 'MgTi2O5(s,l)', 'MgS(s)', 'Mg2SiO4(s,l)', 'Mg2TiO4(s,l)', 'Mg2Si(s,l)', 'Mg3N2(s)', 'Mg3P2O8(s,l)', 'Mn(s,l)', 'TiN(s,l)', 'VN(s)', 'N2O4(s,l)', 'Si3N4(s)', 'P3N5(s)', 'Na(s,l)', 'NaO2(s)', 'Na2O(s,l)', 'Na2O2(s)', 'Na2SiO3(s,l)', 'Na2SO4(s,l)', 'Na2Si2O5(s,l)', 'Na2S(s,l)', 'Na2S2(s,l)', 'Ni(s,l)', 'NiS(s,l)', 'NiS2(s,l)', 'Ni3S2(s,l)', 'Ni3S4(s)', 'TiO(s,l)', 'VO(s,l)', 'SiO2(s,l)', 'TiO2(s,l)', 'Ti2O3(s,l)', 'V2O3(s,l)', 'ZnSO4(s)', 'V2O4(s,l)', 'Ti3O5(s,l)', 'V2O5(s,l)', 'Ti4O7(s,l)', '(P2O5)2(s)', 'P(s,l)', 'P4S3(s,l)', 'S(s,l)', 'SiS2(s,l)', 'Si(s,l)', 'Ti(s,l)', 'V(s,l)', 'Zn(s,l)', 'CaTiO3(s)', 'MnS(s)', 'CaSiO3(s)', 'MnSiO3(s)', 'Ca2SiO4(s)', 'Fe2SiO4(s)', 'CaMgSi2O6(s)', 'Ca2Al2SiO7(s)', 'CaAl2Si2O8(s)', 'KAlSi3O8(s)', 'NaAlSi3O8(s)', 'H2O(s,l)', 'CH4(s,l)', 'N2(s,l)', 'NH3(s,l)', 'CO2(s,l)', 'CO(l)', 'SiO(s)']
list_of_elements = [
    "Al", "Ar", "C", "Ca", "Cl", "Co", "Cr", "Cu", "F", "Fe", "Ge",
    "H", "He", "K", "Mg", "Mn", "N", "Na", "Ne", "Ni", "O", "P",
    "S", "Si", "Ti", "V", "Zn"
]

output = '../output'
matfiles = glob.glob(os.path.join(output, '*.mat'))

#%% Parameters
#c_to_o = 0.1
#metallicity = 0.01

c_to_os = [0.1, 0.3, 0.55, 0.75, 1, 1.25, 1.5]
metallicities = [0.01, 0.1, 1, 10, 100]

for c_to_o in c_to_os:
    for metallicity in metallicities:

        CO = f'C/O: {c_to_o}'
        FEH = f'Fe/H (times Solar): {metallicity}'

        #%% Element class for each element
        
        @dataclass
        class ElementInfo:
            condensates: list[str] = field(default_factory=list)
            abundance: float      = None
            counts: list[int]     = field(default_factory=list)
        
        element_dict = { el: ElementInfo() for el in list_of_elements }
        
        # populate
        for cond in list_of_condensates:
            # strip off the “(s,l)” part
            formula = cond.split('(')[0]
            # grab all element symbols in the formula
            tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        
            # build a quick lookup: { "V":2, "O":5, ... }
            formula_counts = { sym: int(num) if num else 1 for sym, num in tokens }
        
            # for each element present, record both the condensate and its count
            for el in list_of_elements:
                if el in formula_counts:
                    info = element_dict[el]
                    info.condensates.append(cond)
                    info.counts.append(formula_counts[el])
        
        
        #%% Load data
        
        # Filename pattern for identifying CO/FEH
        pattern = r'CO(?P<c_to_o>[^_]+)_M(?P<metallicity>[^_]+)_full\.mat'
        
        # Store Matlab data files as a double-dictionary
        mds = defaultdict(lambda: defaultdict(dict))
        for matfile in matfiles:
            # Identify C/O and Fe/H
            match = re.search(pattern, matfile)
        
            if match:
                s_c_to_o = f'C/O: {match.group("c_to_o")}'
                s_metallicity = f'Fe/H (times Solar): {match.group("metallicity")}'
            else:
                raise ValueError(f'Error reading {matfile}.')
            
            md = matfile
        
            # Read values from file (inconsistent)
            #c_to_o = f"C/O: {md['c_to_o'][0,0]}"
            #metallicity = f"Fe/H: {md['metallicity'][0,0]}"
            
            mds[f'{s_metallicity}'][f'{s_c_to_o}'] = md
        
        md = scipy.io.loadmat(mds[f'{FEH}'][f'{CO}'])
        
        #%% Extract data
        
        # Extract Matlab data
        reshaped_num_dens_cond = md['reshaped_num_dens_cond']
        
        Pgrid = md['Pgrid'][:,0]
        #Tgrid = md['Tgrid'][:,0]
        #pressure = md['pressure'][:,0]
        temperature = md['temperature'][:,0]
        
        #T_hi = md['T_hi'][0,0]
        #T_lo = md['T_lo'][0,0]
        #P_hi = md['P_hi'][0,0]
        #P_lo = md['P_lo'][0,0]
        
        num_pres = md['num_pres'][0,0]
        #num_temp = md['num_temp'][0,0]
        num_cond = md['num_cond'][0,0]
        
        #c_to_o = md['c_to_o'][0,0]
        #metallicity = md['metallicity'][0,0]
        #s_to_h = md['s_to_h'][0,0]
        
        #%% Analysis functions
        
        def condensate_point(cutoff):
            '''
            Generates matrix of size (num_pres, num_condensates)
            
            Each column corresponds to each condensate.
            
            Each row corresponds to a single pressure point (row_idx = p_idx).
            
            Each point identifies the highest temperature at which the condensate
            condenses at the corresponding pressure point.
            '''
            
            results = []
            numdensities = []
            
            for cond_idx in range(num_cond):
                
                condensate_pts = []
                num_dens_pts = []
                
                for p_idx in range(num_pres):
                    # Extract the temperature column for the current condensate and pressure
                    temp_column = reshaped_num_dens_cond[p_idx, :, cond_idx]
                    
                    # Find the index of the first non-zero temperature
                    non_zero_indices = np.nonzero(temp_column)[0]
                    if non_zero_indices.size > 0:
                        # Append to list only if above cutoff number density
                        j = 0
                        for t_idx in non_zero_indices:
                            if reshaped_num_dens_cond[p_idx, t_idx, cond_idx] >= cutoff:
                                condensate_pts.append(temperature[t_idx])
                                num_dens_pts.append(reshaped_num_dens_cond[p_idx, t_idx, cond_idx])
                                j+=1
                                break
                        if j == 0:
                            # No points above cutoff detected
                            condensate_pts.append(np.nan)
                            num_dens_pts.append(np.nan)
                    else:
                        # Handle cases where all temperatures are zero for this pressure and condensate
                        condensate_pts.append(np.nan)
                        num_dens_pts.append(np.nan)
                
                # List >> Array
                condensate_pts = np.array(condensate_pts)
                num_dens_pts = np.array(num_dens_pts)
                
                # Create list of arrays
                results.append(condensate_pts)
                numdensities.append(num_dens_pts)
            
            # List of arrays >> Matrix
            results = np.stack(results, axis=1)
            numdensities = np.stack(numdensities, axis=1)
            
            return results, numdensities
        
        results, numdensities = condensate_point(0)
        
        def find(symbol):
            for i, condensate in enumerate(list_of_condensates):
                # Remove the phase part (e.g., '(s)', '(s,l)')
                base_name = condensate.split('(s')[0].strip()
                if symbol == base_name:
                    data = results[:,i]
                    nd = numdensities[:,i]
                    return [data, condensate, nd]
                elif symbol == condensate:
                    data = results[:,i]
                    nd = numdensities[:,i]
                    return [data, condensate, nd]
        
            # If no match is found:
            print(f'{symbol} not found within FastChem! Skipping.')
            return [np.full(len(results[:,0]),np.nan), None, None]
        
        def abundance(fastchem, c_to_o, metallicity, s_to_h=1.0):
            '''
            From Sander Somers (Feb 2025)
            
            Changes the elemental abundace for custom C/O, metallicity, or S/H
            '''
            solar_abundances = np.array(fastchem.getElementAbundances())
            
            element_abundances = np.copy(solar_abundances)  # Create array of abundances to be modified
            
            index_C = fastchem.getElementIndex('C')
            index_O = fastchem.getElementIndex('O')
            index_S = fastchem.getElementIndex('S')
            
            if metallicity != 1:
                for k in range(0, fastchem.getElementNumber()):
                    if fastchem.getElementSymbol(k) != 'H' and fastchem.getElementSymbol(k) != 'He' and fastchem.getElementSymbol(k) != 'S':
                        element_abundances[k] *= metallicity  # Set non-sulphur, non-H & non-He abundances to desired metallicity
                    elif fastchem.getElementSymbol(k) == 'S':
                        element_abundances[k] *= metallicity * s_to_h  # Set sulphur abundance to desired metallicity * enhancement
        
            elif metallicity == 1:  # As don't need to perform *Z, so only have to amend sulphur enhancement
                    element_abundances[index_S] *= s_to_h
            
            # Setting the C and O abundances as a function of the C/O ratio, conserving total metallicity   
            m_init = element_abundances[index_C] + element_abundances[index_O]  # Initial abundance of C + O
            element_abundances[index_C] = element_abundances[index_O] * c_to_o  # New abundance of C as determined by C/O * abundance(O)
            m_fin = element_abundances[index_C] + element_abundances[index_O]  # New total abundance; abundance(O) + new_abundance(C)
            m_cor = m_init/m_fin  # Calculating correcting factor required to correct new total abundance to be equal to initial abundance
            element_abundances[index_C]*=m_cor  # Applying factor to C abundance to obtain final C abundance
            element_abundances[index_O]*=m_cor  # Applying factor to O abundance to obtain final O abundance
            
            fastchem.setElementAbundances(element_abundances)
            
            return fastchem
        
        #%% Pull out abundances
        
        fastchem = pyfastchem.FastChem(
          '../input/element_abundances/asplund_2009.dat', 
          '../input/logK/logK.dat',
          '../input/logK/logK_condensates.dat',
          1)
        
        fastchem = abundance(fastchem, c_to_o, metallicity)
        abundances = np.array(fastchem.getElementAbundances())[1:]
        abundances = abundances/np.min(abundances) # Normalize to minimum = 1
        
        for i, a in enumerate(abundances):
            element_dict[list_of_elements[i]].abundance = a
        
        #%% Primary condensate identification
        
        pConds = []
        for i, p in enumerate(Pgrid):
            slice_elem_dict = copy.deepcopy(element_dict)
            primary_condensates = copy.deepcopy(list_of_condensates)
            
            slice = results[i,:]
            slice = np.nan_to_num(slice, nan=-1)
            idxs = np.argsort(slice)[::-1]
            for idx in idxs:
                cond = list_of_condensates[idx]
                
                # strip off the “(s,l)” part
                formula = cond.split('(s')[0]
                # grab all element symbols in the formula
                tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
                # build a quick lookup: { "V":2, "O":5, ... }
                formula_counts = { sym: int(num) if num else 1 for sym, num in tokens }
                
                # skip if any element is already exhausted
                if any(slice_elem_dict[el].abundance <= 1 for el in formula_counts):
                    primary_condensates.remove(cond)
                    continue
                
                # limiting = one with smallest abundance / stoich_count
                limiting_el = min(formula_counts,
                    key=lambda el: slice_elem_dict[el].abundance / formula_counts[el]
                )
                # how much “reaction” we can do
                extent = slice_elem_dict[limiting_el].abundance / formula_counts[limiting_el]
        
                # subtract from every element: extent x its stoich count
                for el, count in formula_counts.items():
                    slice_elem_dict[el].abundance -= extent * count
            
            pConds.append(primary_condensates)
        
        #%% Reduction
            
        pCond_red = []
        for slice in pConds:
            for cond in slice:
                if cond not in pCond_red:
                    pCond_red.append(cond)
        
        print(f'Primary Condensates for ((CO {c_to_o} | FeH {metallicity})):')
        print(pCond_red)
        print('\n')

