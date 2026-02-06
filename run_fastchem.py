#%% Boot

import pyfastchem
#from save_output import saveChemistryOutput, saveCondOutput, saveMonitorOutput, saveChemistryOutputPandas, saveMonitorOutputPandas, saveCondOutputPandas
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import scipy.io

# if True, read in .mat data from declared source. Replaces FastChem simulation.
readMAT = False

if readMAT:
    # Change this
    matfile = '../output/Asplund2009_CO0.55_M1_full.mat'

if not readMAT:
    # File output name, change this
    filetag = 'NewGraphite'
    file_destination = '../output/NewGraphite'

# Late-M Dwarf: Ludwig, F. Allard, P.H. Hauschildt (2002)
dwarf = np.loadtxt('../input/example_p_t_structures/Late_M-dwarf.dat')

# if True, augment FastChem solar abundances with custom C/O / Metallicity / S/H
# Does NOT override readMAT
changeAbundances = True

if readMAT == True:
    changeAbundances = False

if not changeAbundances:
    c_to_os = [0.55] # absolute ratio (0.55 = solar)
    metallicities = [1.0] # times solar metallicity
    s_to_h = 1.0 # Sulphur abundance relative to Z
else:
    #metallicities = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100]
    #c_to_os = [0.10, 0.20, 0.30, 0.40, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]
    metallicities = [100]
    c_to_os = [1.00, 1.60]
    s_to_h = 1.0

#%%

for metallicity in metallicities:
    for c_to_o in c_to_os:
        
        print(f'Processing Fe/H = {metallicity} | C/O = {c_to_o}...')
        print('############################################')

        #%% Input parameters
        if not readMAT:
            # Declare parameters for creating p-T structure
            T_hi = 6000 # K
            T_lo = 100  # K
            P_hi = 3    # 1e{P_hi}
            P_lo = -6   # 1e{P_lo}
            
            num_temp = 500 # num of temperature points
            num_pres = 100 # num of pressure points
        
        else:
            # Read in data
            md = scipy.io.loadmat(matfile)
            
            reshaped_num_dens_cond = md['reshaped_num_dens_cond']
            
            try:
                list_of_condensates = md['list_of_condensates']
            except:
                # If list not present within .mat, populate with default conds
                list_of_condensates = ['Al(s)', 'AlClO(s)', 'AlCl3(s,l)', 'KAlCl4(s)', 'NaAlCl4(s)', 'K3AlCl6(s)', 'Na3AlCl6(s)', 'AlF3(s,l)', 'K3AlF6(s)', 'Na3AlF6(s,l)', 'AlN(s)', 'NaAlO2(s)', 'K3Al2Cl9(s)', 'MgAl2O4(s,l)', 'Al2O3(s,l)', 'Al2SiO5(s)', 'Al2S3(s)', 'Na5Al3F14(s,l)', 'Al6Si2O13(s)', 'C(s)', 'CuCN(s)', 'KCN(s,l)', 'K2CO3(s,l)', 'MgCO3(s)', 'NaCN(s,l)', 'NaCO3(s,l)', 'SiC(s)', 'TiC(s,l)', 'Cr3C2(s)', 'MgC2(s)', 'Al4C3(s)', 'Cr7C3(s)', 'Mg2C3(s)', 'Ni(CO)4(l)', 'Fe(CO)5(l)', 'Cr23C6(s)', 'Ca(s,l)', 'CaCl2(s,l)', 'CaF2(s,l)', 'Ca(OH)2(s)', 'CaO(s,l)', 'CaS(s)', 'CuCl(s,l)', 'NH4Cl(s)', 'NH4ClO4(s)', 'KCl(s,l)', 'KClO4(s)', 'NaCl(s,l)', 'NaClO4(s)', 'CoCl2(s,l)', 'CuCl2(s)', 'FeCl2(s,l)', 'MgCl2(s,l)', 'NiCl2(s,l)', 'SCl2(l)', 'ClSSCl(l)', 'TiCl2(s)', 'FeCl3(s,l)', 'TiCl3(s)', 'TiCl4(s,l)', 'Co(s,l)', 'CoF2(s,l)', 'CoF3(s)', 'CoO(s)', 'CoSO4(s)', 'Co3O4(s)', 'Cr(s,l)', 'CrN(s)', 'Cr2N(s)', 'Cr2O3(s,l)', 'Cu(s,l)', 'CuF(s)', 'CuF2(s,l)', 'Cu(OH)2(s)', 'CuO(s)', 'CuSO4(s)', 'Cu2O(s,l)', 'KF(s,l)', 'NaF(s,l)', 'FeF2(s,l)', 'K(HF2)(s,l)', 'MgF2(s,l)', 'FeF3(s)', 'TiF3(s)', 'TiF4(s)', 'Fe(s,l)', 'Fe(OH)2(s)', 'Fe(OH)3(s)', 'FeO(s,l)', 'FeSO4(s)', 'FeS(s,l)', 'FeS2(s)', 'Fe2O3(s)', 'Fe2(SO4)3(s)', 'Fe3O4(s)', 'KH(s)', 'KOH(s,l)', 'NaH(s)', 'NaOH(s,l)', 'MgH2(s)', 'Mg(OH)2(s)', 'O2S(OH)2(s,l)', 'TiH2(s)', 'H3PO4(s,l)', 'N2H4(l)', 'H2SO4.H2O(s,l)', 'H2SO4.2H2O(s,l)', 'H2SO4.3H2O(s,l)', 'H2SO4.4H2O(s,l)', 'K(s,l)', 'KO2(s)', 'K2O(s)', 'K2O2(s)', 'K2SiO3(s,l)', 'K2SO4(s,l)', 'K2S(s,l)', 'Mg(s,l)', 'MgO(s,l)', 'MgSiO3(s,l)', 'MgTiO3(s,l)', 'MgSO4(s,l)', 'MgTi2O5(s,l)', 'MgS(s)', 'Mg2SiO4(s,l)', 'Mg2TiO4(s,l)', 'Mg2Si(s,l)', 'Mg3N2(s)', 'Mg3P2O8(s,l)', 'Mn(s,l)', 'TiN(s,l)', 'VN(s)', 'N2O4(s,l)', 'Si3N4(s)', 'P3N5(s)', 'Na(s,l)', 'NaO2(s)', 'Na2O(s,l)', 'Na2O2(s)', 'Na2SiO3(s,l)', 'Na2SO4(s,l)', 'Na2Si2O5(s,l)', 'Na2S(s,l)', 'Na2S2(s,l)', 'Ni(s,l)', 'NiS(s,l)', 'NiS2(s,l)', 'Ni3S2(s,l)', 'Ni3S4(s)', 'TiO(s,l)', 'VO(s,l)', 'SiO2(s,l)', 'TiO2(s,l)', 'Ti2O3(s,l)', 'V2O3(s,l)', 'ZnSO4(s)', 'V2O4(s,l)', 'Ti3O5(s,l)', 'V2O5(s,l)', 'Ti4O7(s,l)', '(P2O5)2(s)', 'P(s,l)', 'P4S3(s,l)', 'S(s,l)', 'SiS2(s,l)', 'Si(s,l)', 'Ti(s,l)', 'V(s,l)', 'Zn(s,l)', 'CaTiO3(s)', 'MnS(s)', 'CaSiO3(s)', 'MnSiO3(s)', 'Ca2SiO4(s)', 'Fe2SiO4(s)', 'CaMgSi2O6(s)', 'Ca2Al2SiO7(s)', 'CaAl2Si2O8(s)', 'KAlSi3O8(s)', 'NaAlSi3O8(s)', 'H2O(s,l)', 'CH4(s,l)', 'N2(s,l)', 'NH3(s,l)', 'CO2(s,l)', 'CO(l)', 'SiO(s)'] 
            
            Pgrid = md['Pgrid'][:,0]
            Tgrid = md['Tgrid'][:,0]
            pressure = md['pressure'][:,0]
            temperature = md['temperature'][:,0]
            
            T_hi = md['T_hi'][0,0]
            T_lo = md['T_lo'][0,0]
            P_hi = md['P_hi'][0,0]
            P_lo = md['P_lo'][0,0]
            
            num_pres = md['num_pres'][0,0]
            num_temp = md['num_temp'][0,0]
            num_cond = md['num_cond'][0,0]
            
            c_to_o = md['c_to_o'][0,0]
            metallicity = md['metallicity'][0,0]
            s_to_h = md['s_to_h'][0,0]
            
        
        #%% Create p-T structure
        
        def pT_grid(T_hi, T_lo, P_hi, P_lo, num_temp, num_pres):
            '''
            Generates a p-T grid in the form of two arrays usable by FastChem.
            
            Both arrays are of size (num_temp * num_pres), and the indices of each
            correspond to one another.
            
            Temperature array changes fast (first num_temp points correspond to pressure[0])
            
            Pressure array changes slow (first pressure point repeated num_temp times)
        
            Returns
            -------
            pressure : NumPy Array
                Concatenated array of pressure values (slow)
            temperature : NumPy Array
                Concatenated array of temperature values (fast)
            '''
            # Establish p-T grid
            Pgrid = np.logspace(P_hi, P_lo, num_pres)
            Tgrid = np.linspace(T_hi, T_lo, num_temp)
            
            # Modulate arrays for iteration
            T, P = [], []
            for p in Pgrid:
                P.extend(np.full(len(Tgrid), p))
                T.extend(Tgrid)
            pressure = np.array(P)
            temperature = np.array(T)
            
            return pressure, temperature, Pgrid, Tgrid
        
        if not readMAT:
            pressure, temperature, Pgrid, Tgrid = pT_grid(T_hi, T_lo, P_hi, P_lo, num_temp, num_pres)
        
        #%% FastChem Abundance Function
        
        def abundance(fastchem, c_to_o, metallicity, s_to_h):
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
        
        #%% Start FastChem procedure
        
        number_densities_cond_all = []
        
        # Define the directory for the output
        output_dir = '../output'
        
        if not readMAT:
            # Create a FastChem object
            fastchem = pyfastchem.FastChem(
              '../input/element_abundances/asplund_2009.dat', 
              '../input/logK/logK.dat',
              '../input/logK/logK_condensates.dat',
              1)
            
            if changeAbundances:
                fastchem = abundance(fastchem, c_to_o, metallicity, s_to_h)
            
            # Pull out list of condensates
            list_of_condensates = []
            numSpecies = fastchem.getCondSpeciesNumber()
            for i in range(numSpecies):
                symbol = fastchem.getCondSpeciesSymbol(i)
                list_of_condensates.append(symbol)
            
            for p_idx, p in enumerate(Pgrid):
                print(f"Processing pressure {p_idx + 1}/{len(Pgrid)}: {p:.2e} bar")
                
                # Create the input and output structures for FastChem
                input_data = pyfastchem.FastChemInput()
                output_data = pyfastchem.FastChemOutput()
            
                input_data.temperature = Tgrid
                input_data.pressure = np.full(len(Tgrid), p)
            
                # Use rainout condensation approach
                input_data.rainout_condensation = True
                
                # Run FastChem on the entire p-T structure
                fastchem_flag = fastchem.calcDensities(input_data, output_data)
            
                #convergence summary report
                print("FastChem reports:")
                print("  -", pyfastchem.FASTCHEM_MSG[fastchem_flag])
                
                if np.amin(output_data.element_conserved[:]) == 1:
                  print("  - element conservation: ok")
                else:
                  print("  - element conservation: fail")
            
                # Convert the output into a numpy array
                number_densities = np.array(output_data.number_densities)
                number_densities_cond = np.array(output_data.number_densities_cond)
            
                number_densities_cond_all.append(output_data.number_densities_cond.copy())
            
            number_densities_cond_all = np.array(number_densities_cond_all)
        
        #%% Identifying condensation points on p-T grid
        
        if not readMAT:
            num_cond = number_densities_cond_all.shape[2] # Number of condensates
            
            # Reshape number density condensate matrix into easier to acess 3D matrix
            reshaped_num_dens_cond = number_densities_cond_all.reshape(num_pres, num_temp, num_cond)
        
        # Identify 
        
        cutoff = 0 #Number density cutoff (cm^-3)
        
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
        
        results, numdensities = condensate_point(cutoff)
        
        #%% Find Condensate function
        
        def find(symbol, results=results, list_of_condensates=list_of_condensates):
            for i, condensate in enumerate(list_of_condensates):
                # Remove the phase part (e.g., '(s)', '(s,l)')
                base_name = re.sub(r"\((?:s|l|s,l)\)$", "", condensate).strip()
                if symbol == base_name:
                    data = results[:,i]
                    nd = numdensities[:,i]
                    return [data, condensate, nd]
        
            # If no match is found:
            print(f'{symbol} not found within FastChem! Skipping.')
            return [np.full(len(results[:,0]),np.nan), None, None]
        
        #%% Curve-Smoother function
        
        def smooth(list_of_symbols, results=results, list_of_condensates=list_of_condensates,
                   overwrite=False):
            list_of_results, list_of_labels, list_of_nd = [], [], []
            for symbol in list_of_symbols:
                data, label, nd = find(symbol, results, list_of_condensates)
                list_of_results.append(data)
                list_of_labels.append(label)
                list_of_nd.append(nd)
                
            # Stack data into 2D array, create empty result array to store data in
            stacked_data = np.stack(list_of_results, axis=0)
            result = np.full(stacked_data.shape, np.nan)
            
            # Finds the condensate with highest T at each P, stores into result
            
            # # Excerpt from ChatGPT to work around np.nan issues:
            # # # # # # # # # # # # # # # # # # # # # # #
            
            # Identify columns (points) that have at least one valid (non-NaN) value.
            valid_cols = ~np.all(np.isnan(stacked_data), axis=0)
            
            # Prepare an empty array for indices of the maximum values.
            # For valid columns, use np.nanargmax to get the index of the maximum (ignoring NaNs).
            imax = np.empty(stacked_data.shape[1], dtype=int)
            # This works only for columns that are not entirely NaN.
            imax[valid_cols] = np.nanargmax(stacked_data[:, valid_cols], axis=0)
            
            # Get the indices for the valid columns.
            valid_col_indices = np.arange(stacked_data.shape[1])[valid_cols]
            
            # Copy the maximum values from the valid columns into the result array.
            result[imax[valid_cols], valid_col_indices] = stacked_data[imax[valid_cols], valid_col_indices]
            
            # # # # # # # # # # # # # # # # # # # # # # #
            
            # Return results to lists
            list_of_smooth_data = [result[i,:] for i in range(result.shape[0])]
            
            return [list_of_smooth_data, list_of_labels, list_of_nd]
                
        
        #%% Recreating Mbarek Plot 1
        def plot1_unsmooth():
            plt.figure(figsize=(10,8))
            
            # PT Dwarf
            plt.plot(dwarf[1,:], dwarf[0,:],
                     label='Late-M Dwarf TP Profile\nLudwig, F. Allard, P.H. Hauschildt (2002)',
                     linestyle='-.',
                     color='black')
            
            #### Gold
            symbol = 'Fe'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='gold',
                     linestyle='-')
            
            #### Blue
            symbol = 'Ni'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='blue',
                     linestyle='-')
            
            '''
            #### Red
            symbol = 'MgSiO3'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='red',
                     linestyle='--')
            
            symbol = 'Mg2SiO4'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='red',
                     linestyle='-.')
            '''
            
            #### Purple
            symbol = 'VN'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='purple',
                     linestyle='--')
            
            symbol = 'VO'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='purple',
                     linestyle='-')
            
            #### Pink
            symbol = 'SiO2'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='pink',
                     linestyle='-')
            
            #### Lime
            symbol = 'MnS'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='lime',
                     linestyle='-')
            
            #### Brown
            symbol = 'Na2S'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='brown',
                     linestyle='-')
            
            #### Grey
            symbol = 'ZnS'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='grey',
                     linestyle='-')
            
            #### Navy
            symbol = 'KCl'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='navy',
                     linestyle='-')
            
            plt.xlim(350,3000)
            plt.ylim(1e-6,1e3)
            plt.legend()
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Pressure (bar)')
            plt.title(f'Mbarek Plot 1 (Asplund 2009) \n Pressure from 1e{P_hi} to 1e{P_lo} ({num_pres} points) \n Temperature from {T_hi} to {T_lo} ({num_temp} points) \n Cutoff = {cutoff}')
            plt.grid()
            figname = 'Plot1_Unsmooth.png'
            plt.savefig(f'../output/{figname}')
            plt.show()
            
        
        #%% Recreating Mbarek Plot 2
        def plot2_unsmooth():
            plt.figure(figsize=(10,8))
            
            # PT Dwarf
            plt.plot(dwarf[1,:], dwarf[0,:],
                     label='Late-M Dwarf TP Profile\nLudwig, F. Allard, P.H. Hauschildt (2002)',
                     linestyle='-.',
                     color='black')
            
            #### Grey
            symbol = 'Al12CaO19'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='grey',
                     linestyle='-.')
            
            symbol = 'Al4CaO7'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='grey',
                     linestyle='--')
            
            symbol = 'Al2O3'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='grey',
                     linestyle='-')
            
            #### Lime
            symbol = 'TiN'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='lime',
                     linestyle='-')
            
            symbol = 'Ca4Ti3O10'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='lime',
                     linestyle='-')
            
            symbol = 'CaTiO3'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='lime',
                     linestyle='--')
            
            #### Red
            symbol = 'Ca2SiO4'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='red',
                     linestyle='-')
            
            #### Orange
            symbol = 'Mg2SiO4'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='orange',
                     linestyle='-')
            
            #### Yellow
            symbol = 'Cr7C3'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='gold',
                     linestyle='-.')
            
            symbol = 'Cr'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='gold',
                     linestyle='-')
            
            symbol = 'Cr2O3'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='gold',
                     linestyle='--')
            
            plt.xlim(350,3000)
            plt.ylim(1e-6,1e3)
            plt.legend()
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Pressure (bar)')
            plt.title(f'Mbarek Plot 2 (Asplund 2009) \n Pressure from 1e{P_hi} to 1e{P_lo} ({num_pres} points) \n Temperature from {T_hi} to {T_lo} ({num_temp} points) \n Cutoff = {cutoff}')
            plt.grid()
            figname = 'Plot2_Unsmooth.png'
            plt.savefig(f'../output/{figname}')
            plt.savefig(f'../output/Plot2Cutoffs/{figname}')
        plt.show()
        
        
        #%% Smoothed Plot 1
        def plot1_smooth():
            plt.figure(figsize=(7,6))
            
            linestyles = ['-', ':', '--', '-.' ]
            
            # PT Dwarf
            plt.plot(dwarf[1,:], dwarf[0,:],
                     label='Late-M Dwarf TP Profile\nH.-G. Ludwig et al. (2002)',
                     linestyle='-.',
                     color='black')
            
            #### Gold
            symbol = 'Fe'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='gold',
                     linestyle='-')
            
            #### Blue
            symbol = 'Ni'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='blue',
                     linestyle='-')
            
            '''
            #### Red
            list_Si = ['MgSiO3', 'Mg2SiO4']
            ld, ll, lnd = smooth(list_Si)
            for i, data in enumerate(ld):
                plt.plot(data, 
                         Pgrid, 
                         label=ll[i],
                         color='red',
                         linestyle=linestyles[i])
            '''
            
            #### Purple
            list_V = ['VN', 'VO']
            ld, ll, lnd = smooth(list_V)
            for i, data in enumerate(ld):
                plt.plot(data, 
                         Pgrid, 
                         label=ll[i],
                         color='purple',
                         linestyle=linestyles[i])
            '''
            #### Pink
            symbol = 'SiO2'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='pink',
                     linestyle='-')
            '''
            #### Lime
            symbol = 'MnS'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='lime',
                     linestyle='-')
            
            #### Brown
            symbol = 'Na2S'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='brown',
                     linestyle='-')
            
            #### Grey
            symbol = 'ZnS'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='grey',
                     linestyle='-')
            
            #### Navy
            symbol = 'KCl'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='navy',
                     linestyle='-')
            
            plt.xlim(350,3000)
            plt.ylim(1e-6,1e3)
            plt.legend()
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Pressure (bar)')
            plt.title(f'Asplund et al. (2009) Solar Abundances \n C/O = {c_to_o} \n Metallicity = {metallicity}x Solar')
            #plt.title(f'Mbarek Plot 1 (Asplund 2009) \n Smoothed Curves \n Pressure from 1e{P_hi} to 1e{P_lo} ({num_pres} points) \n Temperature from {T_hi} to {T_lo} ({num_temp} points) \n C/O = {c_to_o} \n Metallicity = {metallicity} times Solar')
            plt.grid()
            figname = f'Plot1_Smooth_CO{c_to_o}_M{metallicity}.png'
            plt.savefig(f'../output/{figname}')
            plt.show()
        
        #%% Smoothed Plot 2
        def plot2_smooth():
            plt.figure(figsize=(10,8))
            
            linestyles = ['-', ':', '--', '-.' ]
            
            # PT Dwarf
            plt.plot(dwarf[1,:], dwarf[0,:],
                     label='Late-M Dwarf TP Profile\nLudwig, F. Allard, P.H. Hauschildt (2002)',
                     linestyle='-.',
                     color='black')
            
            #### Grey
            list_Al = ['Al12CaO19', 'Al4CaO7', 'Al2O3']
            ld, ll, lnd = smooth(list_Al)
            for i, data in enumerate(ld):
                plt.plot(data, 
                         Pgrid, 
                         label=ll[i],
                         color='grey',
                         linestyle=linestyles[i])
            
            #### Lime
            list_Ti = ['TiN', 'Ca4Ti3O10', 'CaTiO3']
            ld, ll, lnd = smooth(list_Ti)
            for i, data in enumerate(ld):
                plt.plot(data, 
                         Pgrid, 
                         label=ll[i],
                         color='lime',
                         linestyle=linestyles[i])
            
            #### Red
            symbol = 'Ca2SiO4'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='red',
                     linestyle='-')
            
            #### Orange
            symbol = 'Mg2SiO4'
            data, label, nd = find(symbol)
            plt.plot(data, 
                     Pgrid, 
                     label=label,
                     color='orange',
                     linestyle='-')
            
            #### Yellow
            list_Cr = ['Cr7C3', 'Cr', 'Cr2O3']
            ld, ll, lnd = smooth(list_Cr)
            for i, data in enumerate(ld):
                plt.plot(data, 
                         Pgrid, 
                         label=ll[i],
                         color='gold',
                         linestyle=linestyles[i])
            
            plt.xlim(350,3000)
            plt.ylim(1e-6,1e3)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Pressure (bar)')
            plt.title(f'Mbarek Plot 2 (Asplund 2009) \n Smoothed Curves \n Pressure from 1e{P_hi} to 1e{P_lo} ({num_pres} points) \n Temperature from {T_hi} to {T_lo} ({num_temp} points) \n C/O = {c_to_o} \n Metallicity = {metallicity} times Solar')
            plt.legend()
            plt.grid()
            figname = f'Plot2_Smooth_CO{c_to_o}_M{metallicity}.png'
            plt.savefig(f'../output/{figname}')
            plt.show()
            
        #%% Output (Raw FastChem Data)
        
        #plot1_unsmooth()
        #plot2_unsmooth()
        
        #plot1_smooth()
        #plot2_smooth()
        
        filename = f'{filetag}_CO{c_to_o}_M{metallicity}.mat'
        
# =============================================================================
#         csvfile = f'../output/{filename}_PTCurves.csv'
#         row_header = ['Pressure (bar)'] + list_of_condensates
#         
#         # Combine Pgrid and results side by side.
#         data_full = np.column_stack((Pgrid, results))  # shape becomes (100, 187)
#         
#         with open(csvfile, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(row_header)
#             writer.writerows(data_full)
# =============================================================================
            
        if not readMAT:
            os.makedirs(file_destination, exist_ok=True)
            matfile = os.path.join(file_destination, filename)
            scipy.io.savemat(matfile, mdict={'reshaped_num_dens_cond': reshaped_num_dens_cond,
                                             'list_of_condensates': list_of_condensates,
                                             'Pgrid': Pgrid,
                                             'Tgrid': Tgrid,
                                             'pressure': pressure,
                                             'temperature': temperature,
                                             'num_pres': num_pres,
                                             'num_temp': num_temp,
                                             'num_cond': num_cond, 
                                             'T_hi': T_hi,
                                             'T_lo': T_lo,
                                             'P_hi': P_hi,
                                             'P_lo': P_lo,
                                             'c_to_o': c_to_o,
                                             'metallicity': metallicity,
                                             's_to_h': s_to_h},
                             oned_as='column')
            print(f'FastChem .mat successfully created at "{matfile}".\n')