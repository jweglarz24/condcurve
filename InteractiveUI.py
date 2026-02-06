#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 12:01:06 2025

@author: jweglarz
"""

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import seaborn as sns
import h5py

list_of_elements = [
    "Al", "Ar", "C", "Ca", "Cl", "Co", "Cr", "Cu", "F", "Fe", "Ge",
    "H", "He", "K", "Mg", "Mn", "N", "Na", "Ne", "Ni", "O", "P",
    "S", "Si", "Ti", "V", "Zn"
]

class plotAttributes:
    def __init__(self, CO, FEH, conds, plotFit, limElems):
        self.CO = CO #C/O .hdf5 group string
        self.FEH = FEH #FeH .hdf5 group string
        self.conds = conds #Condensate .hdf5 group strings
        self.plotFit = plotFit #Bool indicating if curve-fit for CondCurve should be plotted
        self.limElems = limElems #Limiting elements .hdf5 attribute strings

file_path = None
plotAttrs = plotAttributes.__new__(plotAttributes)
plotAttrs.conds = set()
plotAttrs.limElems = set()
elem_vars = {}

#%% plotAttribute toggle update functions

def set_CO_label(CO, key):
    # Sets C/O menu labels from .hdf5 info
    def cmd():
        setattr(plotAttrs, 'CO', key)
        mb_CO.config(text=f'C/O = {CO}')
        
        for elem in plotAttrs.limElems:
           toggle_element(elem)
        
    return cmd

def set_FEH_label(FEH, key):
    # Sets Fe/H menu labels from .hdf5 info
    def cmd():
        setattr(plotAttrs, 'FEH', key)
        mb_FEH.config(text=f'Fe/H (times Solar) = {FEH}')
        
        for elem in plotAttrs.limElems:
           toggle_element(elem)
        
    return cmd

def toggle_plotFit():
    if plotFit.get():
        plotAttrs.plotFit = True 
    else:
        plotAttrs.plotFit = False
    
def toggle_element(elem):
    # Add toggled condensates to plotAttrs for plotting
    if elem_vars[elem].get():
        plotAttrs.limElems.add(elem)
    else:
        plotAttrs.limElems.discard(elem)
    
    if not file_path:
        w2.configure(text='No condensates selected.\nPlease load a .hdf5 or .h5 file.')
    elif not hasattr(plotAttrs, 'CO') or not hasattr(plotAttrs, 'FEH'):
        w2.configure(text='No condensates selected.\nPlease select C/O and Fe/H first.')
    else:
        # Clears cond checkbox selections
        plotAttrs.conds.clear()
        
        # Update limElems button label
        label1 = ''
        for elem in sorted(plotAttrs.limElems):
            label1 = label1 + f'{elem} '
        
        label2 = ''
        with h5py.File(file_path) as f:
            cond_groups = f[plotAttrs.CO][plotAttrs.FEH]
            #Check for limiting element attributes in each cond group
            for elem in plotAttrs.limElems:
                for cond in cond_groups:
                    if elem in cond_groups[cond].attrs:
                        # Add toggled condensates to plotAttrs for plotting
                        plotAttrs.conds.add(cond)
                        label2 = label2 + (f'{cond}\n')
            
        w2.config(text=f'Limiting Element(s): {label1}\nSelected Condensates:\n{label2}')

#%% Limiting element menu popup

def open_limElem_menu():
    #Create 200x585 popup
    popup = Toplevel()
    popup.title('Select Limiting Elements')
    popup.geometry('200x585')
    
    #Canvas w/ scrollbar
    canvas = Canvas(popup)
    scrollbar = ttk.Scrollbar(popup, orient='vertical',
                              command=canvas.yview)
    scroll_frame = Frame(canvas)
    
    #Resizes scrollbar as canvas resizes
    scroll_frame.bind("<Configure>",
                      lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    #Anchoring/packing
    canvas.create_window((0,0), window=scroll_frame, anchor='nw')
    canvas.configure(yscrollcommand = scrollbar.set)
    
    w1_elem = Label(scroll_frame, text='Select a limiting element: ')
    w1_elem.pack(side='top', anchor='n')
    
    selectAll = Button(scroll_frame, text='Select All')
    selectAll.pack(side='top', anchor='n')
    
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')
    
    cb_vars = []
    for elem in list_of_elements:
        #Creates or gets BooleanVar for the condensate
        var = elem_vars.get(elem, BooleanVar())
        elem_vars[elem] = var
        
        cb = Checkbutton(scroll_frame, text=elem, variable=var,
                         command=lambda e=elem: toggle_element(e))
        cb_vars.append(var)
        cb.pack(anchor='w')
    
    def select_all():
        if len(plotAttrs.limElems)>0:
            boolean = False
        else:
            boolean = True
        for i, elem in enumerate(list_of_elements):
            cb_vars[i].set(boolean)
            toggle_element(elem)
    
    selectAll.config(command = select_all)
    
#%% .hdf5 file browser

def open_file_dialog():
    
    global file_path
    
    # Function for reading in .hdf5 file from system
    file_path = filedialog.askopenfilename(
            title="Select the .hdf5 file for viewing:",
            initialdir="./", #working dir
            filetypes=(("HDF5 files", "*.hdf5"), ('All files', '*.*'))
            )
    
    if file_path:
        if not file_path.endswith('.hdf5') or file_path.endswith('.h5'):
            w1.config(text=f'Warning: Incorrect file format.')
        else:
            w1.config(text=f'Seleted file:\n{file_path}')
            mb_CO.config(relief=RAISED)
            mb_FEH.config(relief=RAISED)
            
            menu_CO = Menu(mb_CO, tearoff=0)
            menu_FEH = Menu(mb_FEH, tearoff=0)
            with h5py.File(file_path, 'a') as f:
                #Populate C/O menu labels
                for key in f:
                    CO = key.split('_')[1]
                    menu_CO.add_command(label=f'C/O = {CO}',
                                        command=set_CO_label(CO, key))
                #Populate Fe/H menu labels
                first_CO = next(iter(f))
                keys = list(f[first_CO].keys())
                key_idxs = np.argsort([float(key.split('_')[1]) for key in keys])
                for idx in key_idxs:
                    FEH = keys[idx].split('_')[1]
                    menu_FEH.add_command(label=f'Fe/H (times Solar) = {FEH}',
                                         command=set_FEH_label(FEH, keys[idx]))
                    
                '''
                #Populate C/O menu labels
                COs = []
                for key in f:
                    CO = key.split('_')[1]
                    COs.append(CO)
                CO_idx = np.argsort(COs.astype(float))
                for idx in CO_idx:
                    menu_CO.add_command(label=f'C/O = {COs[idx]}',
                                        command=set_CO_label(CO, key))
                
                #Populate Fe/H menu labels
                first_CO = next(iter(f))
                FEHs = []
                for key in f[first_CO]:
                    FEH = key.split('_')[1]
                    FEHs.append(FEH)
                FEH_idx = np.argsort(FEHs.astype(float))
                for idx in FEH_idx:
                    menu_FEH.add_command(label=f'Fe/H (times Solar) = {FEHs[idx]}',
                                         command=set_FEH_label(FEHs[idx], key))
                '''
    
            #Update menu lists
            mb_CO['menu'] = menu_CO
            mb_FEH['menu'] = menu_FEH
            
            for elem in plotAttrs.limElems:
                elem_vars[elem] = BooleanVar(value=False) #Clears limElem checkbox selections
            plotAttrs.limElems.clear() #Clears limElem selections from plotAttrs
            w2.config(text='Selected Condensates:\nNone')
        
    else:
        w1.config(text='No file selected.')
        
#%% Plot function

def plot():
    if not hasattr(plotAttrs, 'CO') or not hasattr(plotAttrs, 'FEH'):
        w3.configure(text='No C/O / Fe/H ratio selected.\nCondensates not plotted.')
        
    elif len(plotAttrs.conds)<1:
        w3.configure(text='Error: No condensates selected.\nPlease select one to plot.')
    
    else:
        w3.configure(text='')
        # Format plot
        fig = Figure(figsize=(8,8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0,3000) #Temp (K)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylim(1e-6,1e3) #Pres (bar)
        ax.set_ylabel('Pressure (bar)')
        ax.invert_yaxis()
        ax.set_yscale('log')
        fig.subplots_adjust(bottom=0.13,top=0.90)
        
        #Set title
        title = f'C/O = {plotAttrs.CO.split("_")[1]} | Fe/H (times Solar) = {plotAttrs.FEH.split("_")[1]}'
        '''
        label = ''
        if len(plotAttrs.limElems)==len(list_of_elements):
            label = '\nAll primary condensates'
        elif plotAttrs.limElems:
            label = label + '\nLimiting Element(s): '
            for elem in plotAttrs.limElems:
                label = label + f'{elem} '
        title = title + label
        '''
        ax.set_title(title)
        
        # Midpoint selector (for annotations)
        def midpoint(arr):
            non_nan_elements = arr[~np.isnan(arr)] #create a new array without NaN values.
            if len(non_nan_elements) == 0:
                return np.nan #return NaN if all elements are NaN.
            else:
                midpoint_index = len(non_nan_elements) // 2 #calculate the midpoint index
                return non_nan_elements[midpoint_index]
        
        # Read button selections, set color/linestyle cycle
        num_colors = len(plotAttrs.limElems)
        colors= sns.color_palette('husl', n_colors=num_colors)
        linestyles = ['-','--','-.',':']
    
        with h5py.File(file_path, 'a') as f:
            condData = f[plotAttrs.CO][plotAttrs.FEH]
            for i, elem in enumerate(plotAttrs.limElems):
                j = 0
                for cond in condData:
                    if elem in condData[cond].attrs:
                        isPrimary = condData[cond]['isPrimary'][()]
                        P = condData[cond]['P'][()]
                        T = condData[cond]['T'][()]
                        
                        Pprim = np.where(isPrimary, P, np.nan)
                        Tprim = np.where(isPrimary, T, np.nan)
                        #nd = condData[cond]['nd'][isPrimary][()]
                        
                        if not np.isnan(T).all():
                            ax.plot(Tprim,Pprim,label=f'{cond}',
                                    color=colors[i],
                                    linestyle=linestyles[j % len(linestyles)])
                            
                            if plotAttrs.plotFit:
                                ax.plot(condData[cond]['fitTemp'][()], Pprim,
                                        linestyle='solid', color='black',
                                        linewidth=0.4)
                            T_ann = midpoint(Tprim)
                            P_ann = midpoint(Pprim)
                            if cond == 'KCl(s,l)       ':
                                ax.annotate(cond.split('(s')[0],
                                        xy     = (T_ann, P_ann),
                                        xytext = (0.86*T_ann, 3*P_ann),
                                        color  = colors[i],
                                        rotation = 90,
                                        fontweight='bold'
                                        )
                            elif cond == 'ZnS(s)         ':
                                ax.annotate(cond.split('(s')[0],
                                        xy     = (T_ann, P_ann),
                                        xytext = (1.03*T_ann, 1.02*P_ann),
                                        color  = colors[i],
                                        rotation = 90,
                                        fontweight='bold'
                                        )
                            else:
                                ax.annotate(cond.split('(s')[0],
                                        xy     = (T_ann, P_ann),
                                        xytext = (1.01*T_ann, 1.02*P_ann),
                                        color  = colors[i],
                                        rotation = 90,
                                        fontweight='bold'
                                        )
                            j = j+1
                        
        ax.grid()
        ax.plot()
        
        # Plot on popup canvas
        #Create 1200x800 popup
        popup = Toplevel()
        popup.title('CondCurve Plot')
        popup.geometry('1200x800')
        
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        
        '''
        # Add labels for fit coefficients
        if plotAttrs.plotFit:
            popup2 = Toplevel()
            popup2.title('Fit Parameters')
            popup2.geometry('800x400')
            
            ws = []
            with h5py.File(file_path, 'a') as f:
                condData = f[plotAttrs.CO][plotAttrs.FEH]
                for cond in plotAttrs.conds:
                    data = condData[cond]
                    
                    if data.attrs['regime'] == '1':
                        coeffs = data['fitCoeffs1'][()]
                    elif data.attrs['regime'] == '2':
                        coeffs = data['fitCoeffs2'][()]
                    elif data.attrs['regime'] == '3':
                        coeffs = data['fitCoeffs3'][()]
                        
                    R2 = data['fitR2'][()][0]
                    
                    if (data.attrs['regime'] == '1' or data.attrs['regime'] == '3'):
                        a1 = coeffs[0]
                        a2 = coeffs[1]
                        a3 = coeffs[2]
                        a4 = coeffs[3]
                        a5 = coeffs[4]
                        
                        ws.append(Label(popup2, text=f'{cond.strip()} fit data:  [a1 = {a1:.3e}, a2 = {a2:.3e}, a3 = {a3:.3e}, a4 = {a4:.3e}, a5 = {a5:.3e}], R2 = {R2:.3f}'))
                    
                    elif data.attrs['regime'] == '2':
                        a1 = coeffs[0]
                        a2 = coeffs[1]
                        a3 = coeffs[2]
                        a4 = coeffs[3]
                        a5 = coeffs[4]
                        
                        ws.append(Label(popup2, text=f'{cond.strip()} fit data:  [a1 = {a1:.3e}, a2 = {a2:.3e}, a3 = {a3:.3e}, a4 = {a4:.3e}, a5 = {a5:.3e}], R2 = {R2:.3f}'))
            for w in ws:
                w.pack(side='top', fill='x')
            '''
        
        toolbar = NavigationToolbar2Tk(canvas,popup) #matplotlib nav toolbar
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')
        canvas.get_tk_widget().pack(side='top',fill='both')
    

#%% tkinter GUI setup

# tkinter root
r = Tk(screenName=None, baseName=None, className='Tk', useTk=1)
r.title('CondCurve Viewer')

# .hdf5 selection label
w1 = Label(r, text='No file selected.')
w1.grid(row=1, column=0, rowspan=1, columnspan=1, padx=10, pady=(0,10))

# C/O menu button
mb_CO = Menubutton(r, text='Select C/O Ratio...',
                    relief=SUNKEN,
                    width=30, height=2)
mb_CO.grid(row=2,column=0, padx=20)

# Fe/H menu button
mb_FEH = Menubutton(r, text='Select Fe/H Ratio...',
                     relief=SUNKEN,
                     width=30, height=2)
mb_FEH.grid(row=3,column=0)

# Limiting element preset selection button
mb_LE = Button(r, text='<< Select Limiting Element Preset >>',
                    command=open_limElem_menu,
                    relief=RAISED,
                    width=28, height=2)
mb_LE.grid(row=5,column=0)

w2 = Label(r, text='Selected Condensates:\nNone')
w2.grid(row=6, column=0)

plotButton = Button(r, text='Plot',
                    command=plot,
                    width=20, height=2)
plotButton.grid(row=7, column=0, pady=(20,0))

w3 = Label(r, text='')
w3.grid(row=8, column=0)

plotFit = IntVar(value=False)
plotAttrs.plotFit = False
plotFitButton = Checkbutton(r, text='Plot curve-fits of primary condensates',
                            command=toggle_plotFit,
                            variable=plotFit, onvalue=1, offvalue=0,
                            width=45,height=2)
plotFitButton.grid(row=9,column=0)

file_browse_button = Button(r, text='Browse for HDF5 file', command=open_file_dialog)
file_browse_button.grid(row=0, column=0, rowspan=1, columnspan=1, pady=(10,0))

r.mainloop()
