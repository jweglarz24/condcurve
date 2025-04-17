#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:00:24 2025

@author: jweglarz
"""

#%% Boot

# Default
import numpy as np
#import matplotlib.pyplot as plt
#import csv
import scipy.io
import os
import glob
import re

# Double-dictionary
from collections import defaultdict

# Interactive plotting
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import clear_output, display

pio.renderers.default = 'notebook'

list_of_condensates = ['Al(s)', 'AlClO(s)', 'AlCl3(s,l)', 'KAlCl4(s)', 'NaAlCl4(s)', 'K3AlCl6(s)', 'Na3AlCl6(s)', 'AlF3(s,l)', 'K3AlF6(s)', 'Na3AlF6(s,l)', 'AlN(s)', 'NaAlO2(s)', 'K3Al2Cl9(s)', 'MgAl2O4(s,l)', 'Al2O3(s,l)', 'Al2SiO5(s)', 'Al2S3(s)', 'Na5Al3F14(s,l)', 'Al6Si2O13(s)', 'C(s)', 'CuCN(s)', 'KCN(s,l)', 'K2CO3(s,l)', 'MgCO3(s)', 'NaCN(s,l)', 'NaCO3(s,l)', 'SiC(s)', 'TiC(s,l)', 'Cr3C2(s)', 'MgC2(s)', 'Al4C3(s)', 'Cr7C3(s)', 'Mg2C3(s)', 'Ni(CO)4(l)', 'Fe(CO)5(l)', 'Cr23C6(s)', 'Ca(s,l)', 'CaCl2(s,l)', 'CaF2(s,l)', 'Ca(OH)2(s)', 'CaO(s,l)', 'CaS(s)', 'CuCl(s,l)', 'NH4Cl(s)', 'NH4ClO4(s)', 'KCl(s,l)', 'KClO4(s)', 'NaCl(s,l)', 'NaClO4(s)', 'CoCl2(s,l)', 'CuCl2(s)', 'FeCl2(s,l)', 'MgCl2(s,l)', 'NiCl2(s,l)', 'SCl2(l)', 'ClSSCl(l)', 'TiCl2(s)', 'FeCl3(s,l)', 'TiCl3(s)', 'TiCl4(s,l)', 'Co(s,l)', 'CoF2(s,l)', 'CoF3(s)', 'CoO(s)', 'CoSO4(s)', 'Co3O4(s)', 'Cr(s,l)', 'CrN(s)', 'Cr2N(s)', 'Cr2O3(s,l)', 'Cu(s,l)', 'CuF(s)', 'CuF2(s,l)', 'Cu(OH)2(s)', 'CuO(s)', 'CuSO4(s)', 'Cu2O(s,l)', 'KF(s,l)', 'NaF(s,l)', 'FeF2(s,l)', 'K(HF2)(s,l)', 'MgF2(s,l)', 'FeF3(s)', 'TiF3(s)', 'TiF4(s)', 'Fe(s,l)', 'Fe(OH)2(s)', 'Fe(OH)3(s)', 'FeO(s,l)', 'FeSO4(s)', 'FeS(s,l)', 'FeS2(s)', 'Fe2O3(s)', 'Fe2(SO4)3(s)', 'Fe3O4(s)', 'KH(s)', 'KOH(s,l)', 'NaH(s)', 'NaOH(s,l)', 'MgH2(s)', 'Mg(OH)2(s)', 'O2S(OH)2(s,l)', 'TiH2(s)', 'H3PO4(s,l)', 'N2H4(l)', 'H2SO4.H2O(s,l)', 'H2SO4.2H2O(s,l)', 'H2SO4.3H2O(s,l)', 'H2SO4.4H2O(s,l)', 'K(s,l)', 'KO2(s)', 'K2O(s)', 'K2O2(s)', 'K2SiO3(s,l)', 'K2SO4(s,l)', 'K2S(s,l)', 'Mg(s,l)', 'MgO(s,l)', 'MgSiO3(s,l)', 'MgTiO3(s,l)', 'MgSO4(s,l)', 'MgTi2O5(s,l)', 'MgS(s)', 'Mg2SiO4(s,l)', 'Mg2TiO4(s,l)', 'Mg2Si(s,l)', 'Mg3N2(s)', 'Mg3P2O8(s,l)', 'Mn(s,l)', 'TiN(s,l)', 'VN(s)', 'N2O4(s,l)', 'Si3N4(s)', 'P3N5(s)', 'Na(s,l)', 'NaO2(s)', 'Na2O(s,l)', 'Na2O2(s)', 'Na2SiO3(s,l)', 'Na2SO4(s,l)', 'Na2Si2O5(s,l)', 'Na2S(s,l)', 'Na2S2(s,l)', 'Ni(s,l)', 'NiS(s,l)', 'NiS2(s,l)', 'Ni3S2(s,l)', 'Ni3S4(s)', 'TiO(s,l)', 'VO(s,l)', 'SiO2(s,l)', 'TiO2(s,l)', 'Ti2O3(s,l)', 'V2O3(s,l)', 'ZnSO4(s)', 'V2O4(s,l)', 'Ti3O5(s,l)', 'V2O5(s,l)', 'Ti4O7(s,l)', '(P2O5)2(s)', 'P(s,l)', 'P4S3(s,l)', 'S(s,l)', 'SiS2(s,l)', 'Si(s,l)', 'Ti(s,l)', 'V(s,l)', 'Zn(s,l)', 'CaTiO3(s)', 'MnS(s)', 'CaSiO3(s)', 'MnSiO3(s)', 'Ca2SiO4(s)', 'Fe2SiO4(s)', 'CaMgSi2O6(s)', 'Ca2Al2SiO7(s)', 'CaAl2Si2O8(s)', 'KAlSi3O8(s)', 'NaAlSi3O8(s)', 'H2O(s,l)', 'CH4(s,l)', 'N2(s,l)', 'NH3(s,l)', 'CO2(s,l)', 'CO(l)', 'SiO(s)']

output = '../output'
matfiles = glob.glob(os.path.join(output, '*.mat'))

#%% Load data

# Filename pattern for identifying CO/FEH
pattern = r'CO(?P<c_to_o>[^_]+)_M(?P<metallicity>[^_]+)_full\.mat'

# Store Matlab data files as a double-dictionary
mds = defaultdict(lambda: defaultdict(dict))
for matfile in matfiles:
    # Identify C/O and Fe/H
    match = re.search(pattern, matfile)

    if match:
        c_to_o = f'C/O: {match.group("c_to_o")}'
        metallicity = f'Fe/H (times Solar): {match.group("metallicity")}'
    else:
        raise ValueError(f'Error reading {matfile}.')
    
    #md = scipy.io.loadmat(matfile)

    # Read values from file (inconsistent)
    #c_to_o = f"C/O: {md['c_to_o'][0,0]}"
    #metallicity = f"Fe/H: {md['metallicity'][0,0]}"
    
    mds[f'{metallicity}'][f'{c_to_o}'] = matfile

#%% Interactive C/O and Metallicity widgets
co_dropdown = widgets.SelectMultiple(
    options = mds['Fe/H (times Solar): 1'].keys(), 
    description = 'C/O Ratio:'
    )

feh_dropdown = widgets.SelectMultiple(
    options = mds.keys(),
    description = 'Metallicity (times Solar):'
    )

#%% Condensate dropdown w/ search function

# Create searchable multi-select component
search_label = widgets.HTML(value="<b>Search Condensates:</b>")
search_box = widgets.Text(placeholder='Type to filter...')

cond_list = list_of_condensates.copy() # Preserve original list

# Filtered select widget
filtered_select = widgets.SelectMultiple(
    options=cond_list,
    description='',
    disabled=False,
    layout={'width': '100%', 'height': '200px'}
)

# Search functionality
def update_search(*args):
    search_term = search_box.value.lower()
    filtered = [c for c in cond_list 
               if search_term in c.lower()]
    filtered_select.options = filtered

search_box.observe(update_search, 'value')

# Deselect all
select_all_btn = widgets.Button(description='Select All', 
                       layout={'width': '120px'})
deselect_all_btn = widgets.Button(description='Deselect All',
                         layout={'width': '120px'})
def select_all(b):
    filtered_select.value = tuple(filtered_select.options)
select_all_btn.on_click(select_all)
    
def deselect_all(b):
    filtered_select.value = ()
deselect_all_btn.on_click(deselect_all)

# Assemble the widget
condensate_selector = widgets.VBox([
    search_label,
    search_box,
    widgets.VBox([select_all_btn, deselect_all_btn], 
        layout={'flex_flow': 'row wrap'}),
    filtered_select
], layout={'border': '1px solid gray', 'padding': '5px'})

# And finally, the dropdown widget!
elem_dropdown = widgets.VBox([
    widgets.HTML("<b>Condensate Selection:</b>"),
    condensate_selector
])

#%% Preset plot dictionary

PRESETS = {
    "Mbarek Plot 1": [
        'Fe(s,l)', 'Ni(s,l)', 'VN(s)', 'VO(s,l)', 
        'SiO2(s,l)', 'MnS(s)', 'Na2S(s,l)', 'ZnS(s)', 
        'KCl(s,l)'
    ],
    "Mbarek Plot 2": [
        'Al2O3(s,l)', 'TiN(s,l)', 'CaTiO3(s)', 
        'Ca2SiO4(s)', 'Mg2SiO4(s,l)', 'Cr7C3(s)', 
        'Cr(s,l)', 'Cr2O3(s,l)'
    ],
    "Primary Condensates for (C/O: 0.1 | Fe/H: 0.01x)" : ['Al2O3(s,l)', 'C(s)', 'NH4Cl(s)', 'KCl(s,l)', 'Co(s,l)', 'Cr(s,l)', 'Cu(s,l)', 'Fe(s,l)', 'MgO(s,l)', 'TiN(s,l)', 'Na2S(s,l)', 'Ni(s,l)', 'P(s,l)', 'S(s,l)', 'V(s,l)', 'Zn(s,l)', 'MnS(s)', 'Ca2SiO4(s)', 'H2O(s,l)', 'NH3(s,l)', 'SiO(s)', 'CaS(s)', 'VN(s)', 'VO(s,l)', 'TiO(s,l)', 'Ti2O3(s,l)', 'P3N5(s)', 'Mg2SiO4(s,l)', 'Cr2O3(s,l)']
}

# Create preset buttons
preset_buttons = [
    widgets.Button(description=name, layout={'width': '300px'})
    for name in PRESETS.keys()
]
preset_status = widgets.Output()

def apply_preset(b):
    with preset_status:
        search_box.value = ''
        preset_status.clear_output()
        target_condensates = PRESETS[b.description]
        
        matched, missing = [], []
        
        for target in target_condensates:
            # Try exact match first
            if target in list_of_condensates:
                matched.append(target)
                continue
            else:
                missing.append(target)
        
        # Update selection
        filtered_select.value = matched
        
        # Print results
        print(f"Applied preset: {b.description}")
        if matched:
            print("\nMatched condensates:\n- " + "\n- ".join(matched))
        if missing:
            print("\nMissing condensates:\n- " + "\n- ".join(missing))

for btn in preset_buttons:
    btn.on_click(apply_preset)
    
# Add presets to UI
preset_box = widgets.VBox([
    widgets.HTML("<b>Preset Selections:</b>"),
    *preset_buttons,
    preset_status], 
    layout={'border': '1px solid gray', 'padding': '5px'})

#%% Load UI
plot_output = widgets.Output()
ui = widgets.VBox([feh_dropdown, co_dropdown, 
                   widgets.HBox([elem_dropdown, preset_box]), 
                   plot_output])

#%% Function for interactive plotting
def plot_mat_data(metallicities, c_to_os, elem_symbols):
    with plot_output:
        # Create figure
        fig = go.Figure()
        fig.data = [] #Clear old traces
        
        for c_to_o in c_to_os:
            for metallicity in metallicities:
                clear_output(wait=True)
                print(f'Selected condensates: {", ".join(elem_symbols)}')
                
                md = scipy.io.loadmat(mds[metallicity][c_to_o])
                
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
        
                # Calculate global color scale limits
                all_nd = []
                valid_condensates = []
                
                # First pass to collect all valid nd values
                for elem_symbol in elem_symbols:
                    data, condensate, nd = find(elem_symbol)
                    if condensate and not np.all(np.isnan(nd)):
                        valid_condensates.append((data, condensate, nd))
                        all_nd.extend(nd[~np.isnan(nd)])
                
                # Set color scale bounds
                cmin, cmax = (np.min(all_nd), np.max(all_nd)) if all_nd else (0, 1)
                
                # Plot
                for elem_symbol in elem_symbols:
                    data, condensate, nd = find(elem_symbol)
        
                    customdata = np.stack((np.full(len(data), 
                                                   condensate + f' ({c_to_o} | {metallicity})'), nd), 
                                          axis=-1)
                    
                    if condensate: #Skip invalid selections
                        fig.add_trace(go.Scatter(
                            x = data, y = Pgrid, mode = 'lines+markers',
                            marker = dict(size=8),
                            name=condensate + f' ({c_to_o} | {metallicity})',  # Empty name to prevent right-side label
                            customdata=customdata,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>" +
                                "Temp: %{x:.2f} K<br>" +
                                "Pressure: %{y:.2e} bar<br>" +
                                "Number Density: %{customdata[1]:.2e} cm⁻³" +
                                "<extra></extra>"  # Removes trace name from hover
                                )))
        
        fig.update_layout(
            title = f'{"Plotting " + ", ".join(elem_symbols)}' + " for solar C:O (0.55), various Metallicity",
            xaxis_title = 'Temperature (k)',
            yaxis_title = 'Pressure (bar)',
            template='plotly',
            yaxis=dict(
                type='log',
                range=[-6,3],
                autorange='reversed'
                ),
            xaxis=dict(
                range=[300, 3000]
                ),
#            coloraxis=dict(
#                colorbar=dict(
#                    title='Number Density (cm⁻³)',
#                    x=1.15  # Move colorbar right
#                )
#            ),
            legend=dict(
               x=0.01,  # Left-align legend
               y=1,
               xanchor='left',
               yanchor='top',
               bgcolor='rgba(0,0,0,0.1)'
               ),
            margin=dict(r=100),
            showlegend=True
            )
        
        pio.show(fig)
        return

#%% Multi-set data plotting function
def plot_multi_dataset(selected_condensates, dataset_configs):
    """
    Plot condensation curves for multiple datasets simultaneously
    
    Parameters:
    -----------
    selected_condensates : list
        List of condensate symbols to plot
    dataset_configs : list of dicts
        Each dict contains 'metallicity', 'c_to_o', and 'style' keys
    """
    with plot_output:
        clear_output(wait=True)
        
        fig = go.Figure()
        
        # Track legend entries to avoid duplicates
        legend_entries = set()
        
        for config in dataset_configs:
            metallicity = config['metallicity']
            c_to_o = config['c_to_o']
            line_style = config.get('style', {})
            
            # Skip if data doesn't exist
            if c_to_o not in mds[metallicity]:
                print(f"No data for {metallicity}, {c_to_o}")
                continue
                
            md = mds[metallicity][c_to_o]
            
            # Extract data similar to plot_mat_data function
            Pgrid = md['Pgrid'][:,0]
            temperature = md['temperature'][:,0]
            # ... other extractions
            
            results, numdensities = condensate_point(0)
            
            # Plot each selected condensate for this dataset
            for condensate in selected_condensates:
                data, cond_name, nd = find(condensate)
                if cond_name:
                    # Create a unique name for the legend that includes dataset info
                    trace_name = f"{cond_name} ({metallicity}, {c_to_o})"
                    
                    # Apply custom styling if provided
                    marker_style = line_style.get('marker', dict(size=8))
                    line_style_dict = line_style.get('line', dict())
                    
                    fig.add_trace(go.Scatter(
                        x=data, 
                        y=Pgrid,
                        mode='lines+markers',
                        name=trace_name,
                        marker=marker_style,
                        line=line_style_dict,
                        customdata=np.stack((np.full(len(data), cond_name), nd), axis=-1),
                        hovertemplate=(
                            "%{customdata[0]}<br>" +
                            "Temp: %{x:.2f} K<br>" +
                            "Pressure: %{y:.2e} bar<br>" +
                            "Number Density: %{customdata[1]:.2e} cm⁻³<br>" +
                            f"C/O: {c_to_o}, Fe/H: {metallicity}x Solar<br>" +
                            "<extra></extra>"
                        )
                    ))
        
        # Update layout with comprehensive title and axes
        fig.update_layout(
            title=f'Condensation Curves Comparison ({len(dataset_configs)} datasets)',
            xaxis_title='Temperature (K)',
            yaxis_title='Pressure (bar)',
            template='plotly',
            yaxis=dict(
                type='log',
                range=[-6,3],
                autorange='reversed'
            ),
            xaxis=dict(
                range=[300, 3000]
            ),
            legend=dict(
                groupclick="toggleitem"
            )
        )
        
        pio.show(fig)
        return

out = widgets.interactive_output(plot_mat_data, {
    'metallicities': feh_dropdown,
    'c_to_os': co_dropdown,
    'elem_symbols': filtered_select
    })

display(ui)