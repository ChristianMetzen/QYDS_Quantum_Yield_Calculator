import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os
import logging

def setup_logger(params):
    
    log_path = os.path.join(params['Results Folder'], params['Basename'] + '.log') 

    logger = logging.getLogger("analysis_logger")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Logging initialized")
    logger.info(f"Log file: {log_path}")

    params['logger'] = logger

    return params

def read_in_UV_Vis_csv(file=None):
    
    if file:
        # Read file
        df = pd.read_csv(file, skip_blank_lines=False)
        
        # Find first empty row
        empty_row_mask = df.replace("", np.nan).isna().all(axis=1)

        if empty_row_mask.any():

            # Get the index of the first True value (first empty row)
            last_row_with_data = empty_row_mask[empty_row_mask].index[0]

            # Slice only the data rows (skip header row 0, stop before empty row)
            df = df.iloc[1:last_row_with_data]
        else:
            df = df.iloc[1:]

        # Shift the column names one to the right
        new_column_names = [None] + list(df.columns[:-1])
        df.columns = new_column_names

        # Delete all wavelenth column except first
        columns_to_delete = [i for i in range(1, len(df.columns)) if i % 2 == 0]
        df = df.drop(columns=df.columns[columns_to_delete])
        df = df.astype(float)
        
        return df
    
    else:

        return
    
def producte_QYDS_dataframe(params):
    
    file = params['QYDS file']

    data = {'Runtime': [], 'Illum Time': [], 'Cell Value': []}

    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            continue   # ignore comment lines

        if line.strip().startswith("Number"):
            continue   # skip header line

        line_data = line.split("\t")

        if len(line_data) >= 7:
            try:
                data['Runtime'].append(float(line_data[3].replace(",", ".")))
                data['Illum Time'].append(float(line_data[4].replace(",", ".")))
                data['Cell Value'].append(float(line_data[6].replace(",", ".")))
            except ValueError:
                # skip empty or malformed lines
                continue

    df = pd.DataFrame(data)
    df = df.astype(float)

    if params['Automatic Measurements'] == 'manual':
        params['QYDS df'] = df
    if params['Automatic Measurements'] == 'auto':
        start, stop, step = params['Start'], params['Stop'], params['Step']
        df = df.iloc[0:0]
        new_rows = pd.DataFrame({
                'Runtime': [pd.NA] * ((stop - start) // step + 1),
                'Illum Time': list(range(start, stop + 1, step)),
                'Cell Value': [pd.NA] * ((stop - start) // step + 1)
            })
        df_new = pd.concat([df, new_rows], ignore_index=True)
        params['QYDS df'] = df_new

    return params

def get_correction_factor(params):
    
    # Extract Parameter
    wavelength = params['Irradiation wavelength']
    channel = params['Channel']
    
    data = {
        "wavelength": [
            260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,
            410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,
            560,570,580,590,600,610,620,630,640,650,660
        ],
        "high": [
            10.042,9.892,9.493,8.934,8.301,7.654,7.036,6.467,5.956,5.503,
            5.105,4.758,4.454,4.189,3.958,3.756,3.578,3.423,3.285,3.164,
            3.056,2.960,2.874,2.797,2.727,2.663,2.605,2.552,2.502,2.455,
            2.410,2.367,2.325,2.284,2.244,2.203,2.162,2.121,2.080,2.038,
            1.995
        ],
        "low": [
            0.502,0.495,0.475,0.447,0.415,0.383,0.352,0.323,0.298,0.275,
            0.255,0.238,0.223,0.209,0.198,0.188,0.179,0.171,0.164,0.158,
            0.153,0.148,0.144,0.140,0.136,0.133,0.130,0.128,0.125,0.123,
            0.120,0.118,0.116,0.114,0.112,0.110,0.108,0.106,0.104,0.102,
            0.100
        ]
    }

    correction_factor = np.interp(wavelength, data['wavelength'], data[channel])
    
    return correction_factor

def calculate_photon_flux(params):
    
    print_func = params['Print function']
    wavelength = params['Irradiation wavelength']
    voltage_after_reference = params['Voltage after ref']
    correction_factor = get_correction_factor(params=params)
    volume = params['Volume']
    logger = params['logger']

    photon_flux_per_vol = wavelength * voltage_after_reference * correction_factor / (constants.Planck * constants.speed_of_light * constants.Avogadro * volume) * 1E-9
    photon_flux = photon_flux_per_vol * volume*1E-3
    abs_photon_flux = photon_flux * constants.Avogadro

    print_func(f'Conversion factor = {correction_factor:.2f} mW V⁻¹')
    logger.info(f'Conversion factor = {correction_factor:.2f} mW V-1')
    print_func(r'Photon flux = {:.2e} mol s⁻¹ L⁻¹ / {:.2e} mol s⁻¹ / {:.2e} s⁻¹'.format(photon_flux_per_vol, photon_flux, abs_photon_flux))
    logger.info(f'Photon flux is {photon_flux_per_vol:.4e} mol s-1 L-1'
                f' or {photon_flux:.4e} mol s-1'
                f' or {abs_photon_flux:.4e} s-1')

    params['Photon flux per vol'] = photon_flux_per_vol
    params['Photon flux'] = photon_flux
    params['Abs photon flux'] = abs_photon_flux

    return params

def create_kinetics_df(params):

    if params['Measurement type'] == 'Multiple Spectra':

        df_spectra = read_in_UV_Vis_csv(file=params['UV Vis file Multiple Spectra'])
        producte_QYDS_dataframe(params=params)
        df_QYDS = params['QYDS df']
  
        if len(df_QYDS) != len(df_spectra.columns) - 1:
            print(f'QYDS dataframe ({len(df_QYDS)}) and irradiation times ({len(df_spectra.columns)}) have not the same length!')
            return

        time = df_QYDS.iloc[:, 1].astype(float).to_numpy()

        df_spectra.columns = (
            ['wavelength / nm'] +
            time[: df_spectra.shape[1] - 1].tolist()
        )

        params['Kinetics df'] = df_spectra
        return params

    if params['Measurement type'] == 'Kinetics':

        # Get absorbances
        df = read_in_UV_Vis_csv(file=params['Kinetics file']).T
        absorbance = df.iloc[1, :-1].astype(float).to_numpy()
        # Correct absorbances with initial spectrum
        df_init = read_in_UV_Vis_csv(file=params['UV Vis file start'])
        wl_index_init = (df_init.iloc[:, 0] - params['Irradiation wavelength']).abs().idxmin()
        A_ini = df_init.iloc[wl_index_init,1]
        dA = A_ini - absorbance[0]
        absorbance += dA

        times = df.iloc[0, :-1].astype(float) * 60   # seconds
        
        df_kinetics = pd.DataFrame(
            data=[absorbance],
            columns=times.astype(str),
        )

        df_kinetics.insert(0, "wavelength / nm", params['Irradiation wavelength'])

        params['Kinetics df'] = df_kinetics
        return params

def fit_absorbance_vs_time_first_order(params):

    print_func = params['Print function']
    initial_guess = params['Initial guess kinetics']   
    df_kinetics = params['Kinetics df']
    kinetic_wl = params['Kinetics wavelength']
    wl_index = (df_kinetics.iloc[:, 0] - kinetic_wl).abs().idxmin()
    absorbances = df_kinetics.iloc[wl_index, 1:].to_numpy(dtype=float)
    time = df_kinetics.columns[1:].astype(float).to_numpy()


    def fitting_function(t, A_inf, A_i, k):
        return A_inf + (A_i - A_inf) * np.exp(-np.abs(k) * t) 

    # sensible default p0 if user provided zeros or nonsense
    p0 = list(initial_guess)
    # try to fill reasonable p0 if zeros:
    if p0 is None or len(p0) != 3:
        p0 = [absorbances[-1], absorbances[0], 0.01]
    else:
        # if k is zero or negative or extremely small, give a small positive start
        if p0[2] == 0 or p0[2] is None:
            p0[2] = 0.01
        # if user put commas as string, ensure numeric (you already convert in GUI)
        p0 = [float(p0[0]), float(p0[1]), float(abs(p0[2]))]

    # Bounds: allow A values to be wide, but k must be >= 0
    # Lower bounds: A_inf and A_i could be negative absorbances in worst case -> set to -np.inf or a reasonable negative
    lower = [-np.inf, -np.inf, 0.0]
    upper = [np.inf, np.inf, np.inf]

    # Try the fit, allow more iterations
    try:
        fit_parameters, covariance = curve_fit(
            fitting_function,
            time,
            absorbances,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000
        )
    except Exception as e:
        # Give a helpful error message
        raise RuntimeError(f"fit failed: {e}")

    fit_parameter_errors = np.sqrt(np.diag(covariance))

    # Print results
    print_func(f'A_inf = {fit_parameters[0]:.6f} ± {fit_parameter_errors[0]:.6f}')
    print_func(f'A_i   = {fit_parameters[1]:.6f} ± {fit_parameter_errors[1]:.6f}')
    print_func(f'k     = {fit_parameters[2]:.6f} ± {fit_parameter_errors[2]:.6f} s⁻¹')

    # Save into params
    params['A_inf'] = fit_parameters[0]
    params['A_inf error'] = fit_parameter_errors[0]
    params['A_i'] = fit_parameters[1]
    params['A_i error'] = fit_parameter_errors[1]
    params['k'] = abs(fit_parameters[2])   # ensure positive
    params['k error'] = fit_parameter_errors[2]
    # store whether curve increased or decreased
    params['kinetics model'] = 'increasing' if params['A_inf'] > params['A_i'] else 'decreasing'

    return params

def plotting_spectra(params, ax=None):

    df_kinetics = params['Kinetics df']
    irradiation_times = df_kinetics.columns[1:].astype(float).to_numpy()
    palette_name = params['Color palette']
    ymin = params['Plotting A_min']
    ymax = params['Plotting A_max']
    xmin = params['Plotting wl_min']
    xmax = params['Plotting wl_max']
    fontsize_legend = params['Legend fontsize spectra']

    # df_QYDS = params['QYDS df']
    # irradiation_times = df_QYDS.iloc[:, 1]
    # df_UV_Vis = params['UV Vis df']

    # Create own figure if subplots are OFF
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))

    palette = sns.color_palette(palette_name, n_colors=len(irradiation_times))

    for i, col in enumerate(df_kinetics.columns[1:], start=1):
        x = df_kinetics.iloc[:, 0]
        y = df_kinetics.iloc[:, i]
        time = irradiation_times[i - 1]

        color = palette[i - 1]
        ax.plot(x, y, color=color, label=f"{time} s")

    ax.set_xlabel("wavelength / nm")
    ax.set_ylabel("absorbance")

    if xmin != 0:
        ax.set_xlim(left=xmin)
    if xmax != 0:
        ax.set_xlim(right=xmax)
    if ymin != 0:
        ax.set_ylim(bottom=ymin)
    if ymax != 0:
        ax.set_ylim(top=ymax)
    if fontsize_legend != 0:
        ax.legend(fontsize=fontsize_legend)
    else:
        ax.legend()

    return ax

def plotting_photokinetics(params, ax=None):

    df_kinetics = params['Kinetics df']
    kinetic_wl = params['Kinetics wavelength']
    wl_index = (df_kinetics.iloc[:, 0] - kinetic_wl).abs().idxmin()
    absorbances = df_kinetics.iloc[wl_index, 1:].to_numpy(dtype=float)
    time = df_kinetics.columns[1:].astype(float).to_numpy()

    t_fit = np.linspace(time[0], time[-1],1000)
    xmin = params['Plotting t_min'] 
    xmax = params['Plotting t_max']
    ymin = params['Plotting A_min']
    ymax = params['Plotting A_max']  
    wavelength = params['Kinetics wavelength']
    A_inf = params['A_inf']
    A_i = params['A_i']
    k = params['k']

    if ax is None:
        fig, ax = plt.subplots()

    def fitting_function(t, A_inf, A_i, k):
        return A_inf + (A_i - A_inf) * np.exp(-k * t)


    ax.plot(time, absorbances, 'ko', label=f'absorbance at {wavelength} nm')
    ax.plot(t_fit, fitting_function(t_fit, A_inf, A_i, k),
            linestyle='-', label='fitted function')

    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('absorbance')
    if xmin != 0:
        ax.set_xlim(left=xmin)
    if xmax != 0:
        ax.set_xlim(right=xmax)
    if ymin != 0:
        ax.set_ylim(bottom=ymin)
    if ymax != 0:
        ax.set_ylim(top=ymax)
    ax.legend()

    return ax

def plotting_conc_profiles(params, ax=None):

    df_conc = params['Conc df']
    time = df_conc['time / s']
    c_A = df_conc['c_A / uM']
    c_B = df_conc['c_B / uM']
    t_min = params["Plotting t_min"]
    t_max = params["Plotting t_max"]
    names = params['Names']

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.scatter(time, c_A, color='C0', label = f'{names[0]}')
    ax.scatter(time, c_B, color='C1', label = f'{names[1]}')
    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('concentration / µM')
    if t_min !=0:
        ax.set_xlim(left=t_min)
    if t_max !=0:
        ax.set_xlim(right=t_max)
    ax.legend()

    return ax 

def plotting_ladanyi_fit(params, ax=None):

    V = params['Volume'] * 1E-3 # L
    I0 = params['Abs photon flux'] # s-1
    df_conc = params['Conc df']
    df_abs_coeff = params['Abs coeff df']

    wl_index = (df_abs_coeff.iloc[:, 0] - params['Irradiation wavelength']).abs().argmin()
    e_A = df_abs_coeff.iloc[wl_index,1]
    e_B = df_abs_coeff.iloc[wl_index,2]   

    c_A = df_conc['c_A / uM'].to_numpy() * 1E-6
    c_B = df_conc['c_B / uM'].to_numpy() * 1E-6
    cA0 = c_A[0]
    cB0 = c_B[0]
    times = df_conc['time / s'].to_numpy()

    phi_A_fit = params['Ladanyi QY A->B']
    phi_B_fit = params['Ladanyi QY B->A']

    names = params['Names']

    t_min = params["Plotting t_min"]
    t_max = params["Plotting t_max"]


    def dcA_dt(t, y, phi_A, phi_B):
        cA, cB = y

        A_tot = cA * e_A + cB * e_B   # model absorbance
        if A_tot <= 0:
            return [0.0, 0.0]

        photon_abs_frac = 1 - 10**(-A_tot)
        prefactor = I0 / (V * constants.Avogadro)

        dcA = prefactor * photon_abs_frac / A_tot * (
            phi_B * cB * e_B - phi_A * cA * e_A
        )

        return [dcA, -dcA]


    def model(times, phi_A, phi_B):
        sol = solve_ivp(
            dcA_dt,
            (times[0], times[-1]),
            y0=[cA0, cB0],
            t_eval=times,
            args=(phi_A, phi_B),
            method="LSODA"
        )
        return sol.y[0]  # c_trans(t)

    if ax is None:
        fig, ax = plt.subplots()

    cA_fit = model(times, phi_A_fit, phi_B_fit)

    ax.plot(times, c_A * 1e6, 'ko', label=f'{names[0]}')
    ax.plot(times, cA_fit * 1e6, '-', label='fitted function')

    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('concentration / µM')

    if t_min !=0:
        ax.set_xlim(left=t_min)
    if t_max !=0:
        ax.set_xlim(right=t_max)
    ax.legend()

    return ax 

def save_individual_plot(plot_func, 
                         params,
                         name):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_func(params, ax)
    fig.savefig(os.path.join(params['Results Folder'], params['Basename'] + name +'.svg'),
        bbox_inches="tight"
    )
    plt.close(fig)

def plotting_stadler(params, figure=None):


    if figure is None:
        fig = plt.figure(layout='constrained', figsize=(12, 5))
    else:
        fig = figure
        fig.clf()  

    axs = fig.subplots(1, 2, sharey=True)
    if params['Measurement type'] == 'Multiple Spectra':
        plotting_spectra(params, ax=axs[0])
        plotting_photokinetics(params, ax=axs[1])
    else:
        plotting_photokinetics(params, ax=axs[0])

    return fig

def plotting_ladanyi(params, figure=None):

    if figure is None:
        fig = plt.figure(layout='constrained', figsize=(12, 5))
    else:
        fig = figure
        fig.clf()
    print(params['Results Folder'])
    axs = fig.subplots(1,3, sharey=False)
    if params['Measurement type'] == 'Multiple Spectra':
        plotting_spectra(params, ax=axs[0])
        plotting_conc_profiles(params, ax=axs[1])
        plotting_ladanyi_fit(params, ax=axs[2])
        
        save_individual_plot(plotting_spectra, params, '_spectra')
        save_individual_plot(plotting_conc_profiles, params, '_conc')
        save_individual_plot(plotting_ladanyi_fit, params, '_ladanyi_fit')
    else:
        plotting_conc_profiles(params, ax=axs[0])
        plotting_ladanyi_fit(params, ax=axs[1])

        save_individual_plot(plotting_ladanyi_fit, params, '_ladanyi_fit')
        save_individual_plot(plotting_conc_profiles, params, '_conc')

    fig.savefig(os.path.join(params['Results Folder'], params['Basename'] + '_results.svg'), 
                bbox_inches="tight")

def get_initial_absorbance_at_irradiation_wavelength(params):

    dataframe = params['Kinetics df']
    wavelength = params['Irradiation wavelength']

    # Find the row index closest to the specified wavelength
    closest_row_index = (dataframe.iloc[:, 0] - wavelength).abs().idxmin()
    
    # Extract absorption values at the closest row index
    absorption_values = dataframe.iloc[closest_row_index, 1:].tolist()

    abs_i0 = absorption_values[0]

    return abs_i0

def calculate_QY_stadler(params):

    print_func = params['Print function']
    k_fit = params['k']
    c_0 = params['Concentration']
    photon_flux = params['Photon flux per vol']
    A_i_0 = get_initial_absorbance_at_irradiation_wavelength(params=params)

    QY_stadler = k_fit * c_0 / (photon_flux * (1 - 10**(-A_i_0)))
    
    print_func(f'A(i,0) = {A_i_0:.2f}')
    print_func('quantum yield from Stadler = {:.2f}'.format(QY_stadler) )

def calculate_pseudo_total_QY_stadler(params):

    print_func = params['Print function']
    k_fit = params['k']
    photon_flux = params['Photon flux per vol']
    A_i_0 = get_initial_absorbance_at_irradiation_wavelength(params=params)

    photo_kinetic_factor = ( 1 - 10**(-A_i_0) ) / A_i_0
    Q = k_fit / ( 1 * photon_flux * photo_kinetic_factor ) 

    print_func('pseudo total QY = {:.2e} L mol⁻¹ cm⁻¹'.format(Q))

def calculate_concentration_profile(params):

    df_kinetics = params['Kinetics df']
    df_abs_coeff = params['Abs coeff df']

    c = params['Concentration']
    time = df_kinetics.columns[1:].astype(float).to_numpy()

    if params['Measurement type'] == 'Multiple Spectra':

        wl = params['Irradiation wavelength']
        wl_index_abs_coeff = (df_abs_coeff.iloc[:, 0] - wl).abs().argmin()
        wl_index_abs = (df_kinetics.iloc[:, 0] - wl).abs().argmin()
        
        e_A = df_abs_coeff.iloc[wl_index_abs_coeff,1]
        e_B = df_abs_coeff.iloc[wl_index_abs_coeff,2] 

        absorbance = df_kinetics.iloc[wl_index_abs,1:].to_numpy(dtype=float)      

        # Fraction of A
        x_B = (absorbance - c * e_A) / (c * (e_B - e_A))

        # Numerical safety
        x_B = np.clip(x_B, 0, 1)
        x_A = 1 - x_B

        df_concentrations = pd.DataFrame({
            'time / s': time,
            'c_A / uM': x_A * c * 1e6,
            'c_B / uM': x_B * c * 1e6,
        })

        if x_A[0] > x_A[-1]:
            params['Names'] = [f'{df_abs_coeff.columns[1]}', f'{df_abs_coeff.columns[2]}']
            # Create dataframe
            df_concentrations = pd.DataFrame({
                'time / s': time,
                'c_A / uM': x_A * c * 1e6,
                'c_B / uM': x_B * c * 1e6,
            })
        else:
            params['Names'] = [f'{df_abs_coeff.columns[2]}', f'{df_abs_coeff.columns[1]}']
            # Create dataframe
            df_concentrations = pd.DataFrame({
                'time / s': time,
                'c_A / uM': x_B * c * 1e6,
                'c_B / uM': x_A * c * 1e6,
            })
        
        params['Conc df'] = df_concentrations

        df_concentrations.to_csv(os.path.join(params['Results Folder'], f'{params['Basename']}_conc.csv'), index=False)
        
        return params

    if params['Measurement type'] == 'Kinetics':

        kinetic_wl = params['Irradiation wavelength']

        # Get absorbances
        absorbance = df_kinetics.iloc[0,1:].to_numpy(dtype=float) 

        # Get absorption coefficients
        wl_index = (df_abs_coeff.iloc[:, 0] - kinetic_wl).abs().argmin()
        e_A = df_abs_coeff.iloc[wl_index,1]
        e_B = df_abs_coeff.iloc[wl_index,2]

        # Fraction of A
        x_B = (absorbance - c * e_A) / (c * (e_B - e_A))

        # Numerical safety
        x_B = np.clip(x_B, 0, 1)
        x_A = 1 - x_B

        df_concentrations = pd.DataFrame({
            'time / s': time,
            'c_A / uM': x_A * c * 1e6,
            'c_B / uM': x_B * c * 1e6,
        })

        if x_A[0] > x_A[-1]:
            params['Names'] = [f'{df_abs_coeff.columns[1]}', f'{df_abs_coeff.columns[2]}']
            # Create dataframe
            df_concentrations = pd.DataFrame({
                'time / s': time,
                'c_A / uM': x_A * c * 1e6,
                'c_B / uM': x_B * c * 1e6,
            })
        else:
            params['Names'] = [f'{df_abs_coeff.columns[2]}', f'{df_abs_coeff.columns[1]}']
            # Create dataframe
            df_concentrations = pd.DataFrame({
                'time / s': time,
                'c_A / uM': x_B * c * 1e6,
                'c_B / uM': x_A * c * 1e6,
            })
        
        params['Conc df'] = df_concentrations
        
        return params

def ladanyi_fit(params=None):

    print_func = params['Print function']
    V = params['Volume'] * 1E-3 # L
    I0 = params['Abs photon flux'] # s-1
    df_conc = params['Conc df']
    df_abs_coeff = params['Abs coeff df']

    wl_index_abs_coeff = (df_abs_coeff.iloc[:, 0] - params['Irradiation wavelength']).abs().argmin()
    e_A = df_abs_coeff.iloc[wl_index_abs_coeff,1]
    e_B = df_abs_coeff.iloc[wl_index_abs_coeff,2] 

    c_A = df_conc['c_A / uM'].to_numpy() * 1E-6
    c_B = df_conc['c_B / uM'].to_numpy() * 1E-6
    cA0 = c_A[0]
    cB0 = c_B[0]
    times = df_conc['time / s'].to_numpy()

    names = params['Names']
    logger = params['logger']

    # Pre Fitting
    # Finite differences
    dc_A = np.diff(c_A)
    dt = np.diff(times)
    dQ = I0 * dt
    
    # Means between points
    c_A_m = 0.5 * (c_A[:-1] + c_A[1:])
    c_B_m = 0.5 * (c_B[:-1] + c_B[1:])
    A_m = e_A * c_A_m + e_B * c_B_m

    # Left-hand side
    y = V * dc_A / dQ

    # Photon absorption factor
    alpha = (1 / constants.Avogadro) * (1 - 10**(-A_m)) / A_m

    # Design matrix
    X = np.column_stack([
        alpha * c_B_m   * e_B,
        -alpha * c_A_m * e_A
    ])

    # Least squares estimate
    Phi_est, *_ = np.linalg.lstsq(X, y, rcond=None)

    phi_cis_to_trans_init = Phi_est[0]
    phi_trans_to_cis_init = Phi_est[1]
    
    initial_guess = [phi_trans_to_cis_init, phi_cis_to_trans_init ]

    def dcA_dt(t, y, phi_A, phi_B):
        cA, cB = y

        A_tot = cA * e_A + cB * e_B   # model absorbance
        if A_tot <= 0:
            return [0.0, 0.0]

        photon_abs_frac = 1 - 10**(-A_tot)
        prefactor = I0 / (V * constants.Avogadro)

        dcA = prefactor * photon_abs_frac / A_tot * (
            phi_B * cB * e_B - phi_A * cA * e_A
        )

        return [dcA, -dcA]


    def model(times, phi_A, phi_B):
        sol = solve_ivp(
            dcA_dt,
            (times[0], times[-1]),
            y0=[cA0, cB0],
            t_eval=times,
            args=(phi_A, phi_B),
            method="LSODA"
        )
        return sol.y[0]  # c_trans(t)


    popt, pcov = curve_fit(
        model,
        times,
        c_A,
        p0=initial_guess,
        # bounds=(0,1)
    )

    phi_A_fit, phi_B_fit = popt

    print_func(f"Φ({names[0]}→{names[1]}) = {phi_A_fit:.4f}")
    print_func(f"Φ({names[1]}→{names[0]}) = {phi_B_fit:.4f}")

    logger.info(f'The {names[0]} to {names[1]} quantum yield is {phi_A_fit:.4f}'
                f' and the {names[1]} to {names[0]} quantum yield {phi_B_fit:.4f}')

    params['Ladanyi QY A->B'] = phi_A_fit
    params['Ladanyi QY B->A'] = phi_B_fit

    cA_fit = model(times, phi_A_fit, phi_B_fit)
    df_ladanyi = pd.DataFrame({
        'time / s' : times,
        'cA exp' : c_A,
        'cA fit' : cA_fit
    })
    df_ladanyi.to_csv(os.path.join(params['Results Folder'], f'{params['Basename']}_ladanyi_fit.csv'), index=False)
        

    return params

def stadler_analysis(params=None):

    create_kinetics_df(params=params)

    calculate_photon_flux(params=params)

    fit_absorbance_vs_time_first_order(params=params)

    if params['Stadler QY'] == 'QY':
        calculate_QY_stadler(params=params)
    if params['Stadler QY'] == 'pseudo total QY':
        calculate_pseudo_total_QY_stadler(params=params)

    return params

def ladanyi_analysis(params=None):
    
    setup_logger(params)
    logger = params['logger']
    logger.info('Started Ladanyi Analyis')
    logger.info(f'Measurement type is {params['Measurement type']}')
    if params['Measurement type'] == "Multiple Spectra":
        logger.info(f'The measurement is perfomed in {params['Automatic Measurements']} mode')
        if params['Automatic Measurements'] == 'auto':
            logger.info(f'Spectra were recorded from {params['Start']} s' 
                        f' to {params['Stop']} s irradiation'
                        f' in {params['Step']} s steps')
    logger.info(f'Irradiation was at {params['Irradiation wavelength']} nm'
                f' in the {params['Channel']} channel'
                f' with {params['Voltage after ref']} V behind the reference cuvette'
                )
    logger.info(f'The volume was {params['Volume']} mL'
                f' and the concentration {params['Concentration']} M')
    
    calculate_photon_flux(params=params)
    create_kinetics_df(params=params)

    df_abs_coeff = pd.read_csv(params['Abs coeff file'])
    params['Abs coeff df'] = df_abs_coeff

    calculate_concentration_profile(params=params)
    ladanyi_fit(params=params)

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="lightyellow",
                         relief="solid", borderwidth=1, font=("Arial", 9))
        label.pack(ipadx=1)

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def run_gui():
    root = tk.Tk()
    root.title("UV-Vis & QYDS Analysis GUI")
    root.state('zoomed') 
    
    # Configure grid
    root.grid_columnconfigure(0, weight=0)  # Left column fixed
    root.grid_columnconfigure(1, weight=1)  # Right column expands
    root.grid_rowconfigure(0, weight=1)     # Allow vertical expansion

    # ========== LEFT COLUMN ==========
    left_frame = ttk.Frame(root, padding=10)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
    left_frame.grid_columnconfigure(0, weight=1)
    left_frame.grid_rowconfigure(2, weight=1)  # Allow method params to expand

    # Title
    title_label = ttk.Label(
        left_frame,
        text="Select QY Calculation",
        font=("TkDefaultFont", 14, "bold")
    )
    title_label.grid(row=0, column=0, pady=(0, 10), sticky="w")

    # Method Selection Tabs
    notebook = ttk.Notebook(left_frame)
    notebook.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    notebook.enable_traversal()

    tab_ladanyi = ttk.Frame(notebook)
    tab_stadler = ttk.Frame(notebook)

    notebook.add(tab_ladanyi, text="Ladanyi")
    notebook.add(tab_stadler, text="Stadler")
    
    # Method-specific content container
    method_param_area = ttk.Frame(left_frame)
    method_param_area.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
    method_param_area.grid_columnconfigure(0, weight=1)

    # File Selection
    file_frame = ttk.LabelFrame(method_param_area, text="File Selection", padding=8)
    file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    file_frame.grid_columnconfigure(0, weight=1)

    def choose_file(var):
        file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file:
            var.set(file)

    def choose_folder(var):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    file_UV_Vis_mult_spectra_var = tk.StringVar(value=r'C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\example_files\Multiple_spectra_manual\260122_azobenzene_40uM_MeOH_325nm_spectra_series.csv')
    file_UV_Vis_start = tk.StringVar(value=r'')
    file_QYDS_var = tk.StringVar(value=r'C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\example_files\Multiple_spectra_manual\260123_325nm_spectra_series.dat')
    file_kinetics = tk.StringVar(value=r'')
    results_folder_var = tk.StringVar(value=None)
    

    ttk.Label(file_frame, text="For Multiple Spectra Measurements: UV Vis file").grid(row=0, column=0, sticky="w")
    entry_frame1 = ttk.Frame(file_frame)
    entry_frame1.grid(row=1, column=0, sticky="ew", pady=2)
    entry_frame1.grid_columnconfigure(0, weight=1)
    ttk.Entry(entry_frame1, textvariable=file_UV_Vis_mult_spectra_var).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(entry_frame1, text="Select UV Vis file", command=lambda: choose_file(file_UV_Vis_mult_spectra_var)).grid(row=0, column=1)

    ttk.Label(file_frame, text="For Multiple Spectra Measurements: QYDS file:").grid(row=2, column=0, sticky="w", pady=(6,0))
    entry_frame2 = ttk.Frame(file_frame)
    entry_frame2.grid(row=3, column=0, sticky="ew", pady=2)
    entry_frame2.grid_columnconfigure(0, weight=1)
    ttk.Entry(entry_frame2, textvariable=file_QYDS_var).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(entry_frame2, text="Select QYDS file", command=lambda: choose_file(file_QYDS_var)).grid(row=0, column=1)

    ttk.Label(file_frame, text="For Kinetics: Kinetic file:").grid(row=4, column=0, sticky="w", pady=(6,0))
    entry_frame3 = ttk.Frame(file_frame)
    entry_frame3.grid(row=5, column=0, sticky="ew", pady=2)
    entry_frame3.grid_columnconfigure(0, weight=1)
    ttk.Entry(entry_frame3, textvariable=file_kinetics).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(entry_frame3, text="Select kinetics file", command=lambda: choose_file(file_kinetics)).grid(row=0, column=1)

    ttk.Label(file_frame, text="For Kinetics: UV Vis file of initital sample").grid(row=6, column=0, sticky="w")
    entry_frame1 = ttk.Frame(file_frame)
    entry_frame1.grid(row=7, column=0, sticky="ew", pady=2)
    entry_frame1.grid_columnconfigure(0, weight=1)
    ttk.Entry(entry_frame1, textvariable=file_UV_Vis_start).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(entry_frame1, text="Select initial UV Vis", command=lambda: choose_file(file_UV_Vis_start)).grid(row=0, column=1)

    ttk.Label(file_frame, text="Results folder:").grid(row=8, column=0, sticky="w")
    entry_frame4 = ttk.Frame(file_frame)
    entry_frame4.grid(row=9, column=0, sticky="ew", pady=2)
    entry_frame4.grid_columnconfigure(0, weight=1)
    ttk.Entry(entry_frame4, textvariable=results_folder_var).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(entry_frame4, text="Select results folder", command=lambda: choose_folder(results_folder_var)).grid(row=0, column=1)


    # Global Measurement Parameters
    meas_frame = ttk.LabelFrame(method_param_area, text="Global Parameters", padding=8)
    meas_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    meas_frame.grid_columnconfigure(1, weight=1)

    volume_var = tk.StringVar(value='3')
    conc_var = tk.StringVar(value='40')
    irradiation_var = tk.StringVar(value='325')
    voltage_var = tk.StringVar(value='7.465')
    start_var = tk.StringVar(value='0')
    stop_var = tk.StringVar(value='0')
    step_var = tk.StringVar(value='0')

    ttk.Label(meas_frame, text="Channel:").grid(row=0, column=0, sticky="w", pady=2, padx=(2,6))
    channel_var = tk.StringVar()
    ttk.Combobox(
        meas_frame,
        textvariable=channel_var,
        values=["high", "low"],
        state="readonly"
    ).grid(row=0, column=1, pady=2, padx=5, sticky="ew")
    channel_var.set("low")  

    ttk.Label(meas_frame, text="Measurement:").grid(row=1, column=0, sticky="w", pady=2, padx=(2,6))
    measurement_var = tk.StringVar()
    ttk.Combobox(
        meas_frame,
        textvariable=measurement_var,
        values=["Multiple Spectra", "Kinetics"],
        state="readonly"
    ).grid(row=1, column=1, pady=2, padx=5, sticky="ew")
    measurement_var.set("Multiple Spectra")  
    
    ttk.Label(meas_frame, text="Manual/Auto:").grid(row=2, column=0, sticky="w", pady=2, padx=(2,6))
    manual_var = tk.StringVar()
    ttk.Combobox(
        meas_frame,
        textvariable=manual_var,
        values=["manual", "auto"],
        state="readonly"
    ).grid(row=2, column=1, pady=2, padx=5, sticky="ew")
    manual_var.set("manual")  

    meas_params = [
        ("Irradiation wavelength / nm", irradiation_var),
        ("Voltage after reference cuvette / V", voltage_var),
        ("Volume / mL", volume_var),
        ("Concentration / µM", conc_var),
        ("For automatic spectra: Start / s", start_var),
        ("For automatic spectra: Stop / s", stop_var),
        ("For automatic spectra: Steps / s", step_var)
    ]

    for i, (label, var) in enumerate(meas_params):
        ttk.Label(meas_frame, text=label + ":").grid(row=i+3, column=0, sticky="w", pady=2, padx=(2,6))
        ttk.Entry(meas_frame, textvariable=var, width=18).grid(row=i+3, column=1, pady=2, padx=5, sticky="ew")

    # Plotting parameters
    plot_frame = ttk.LabelFrame(method_param_area, text="Plotting Parameters", padding=10)
    plot_frame.grid(row=2, column=1, sticky="nsew", pady=(0, 10))

    palette_var = tk.StringVar(value='viridis')
    A_min_var = tk.StringVar(value='0')
    A_max_var = tk.StringVar(value='0')
    wl_min_var = tk.StringVar(value='0')
    wl_max_var = tk.StringVar(value='0')
    font_legend_var = tk.StringVar(value='8')
    t_min_var = tk.StringVar(value='0')
    t_max_var = tk.StringVar(value='0')

    plot_params = [
        ("Color palette", palette_var),
        ("A_min", A_min_var),
        ("A_max", A_max_var),
        ("wl_min", wl_min_var),
        ("wl_max", wl_max_var),
        ("Legend fontsize spectra", font_legend_var),
        ("t_min", t_min_var),
        ("t_max", t_max_var),
    ]

    for i, (label, var) in enumerate(plot_params):
        ttk.Label(plot_frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=2)
        ttk.Entry(plot_frame, textvariable=var, width=20).grid(row=i, column=1, pady=2, padx=5)

    # ========== METHOD-SPECIFIC PARAMETERS ==========
    
    # Stadler Parameters
    stadler_frame = ttk.LabelFrame(method_param_area, text="Stadler Parameters", padding=8)
    stadler_frame.grid_columnconfigure(1, weight=1)

    kinetics_var = tk.StringVar(value='350')
    initial_guess_var = tk.StringVar(value='0,0,0')

    ttk.Label(stadler_frame, text="QY/pseudo total QY").grid(row=0, column=0, sticky="w", pady=2, padx=(2,6))
    stadler_QY_var = tk.StringVar()
    ttk.Combobox(
        stadler_frame,
        textvariable=stadler_QY_var,
        values=["QY", "pseudo total QY"],
        state="readonly"
    ).grid(row=0, column=1, pady=2, padx=5, sticky="ew")
    stadler_QY_var.set("pseudo total QY") 

    stadler_params = [
        ('Kinetics wavelength / nm', kinetics_var),
        ("Initial guess kinetics", initial_guess_var)
    ]

    for i, (label, var) in enumerate(stadler_params):
        ttk.Label(stadler_frame, text=label + ":").grid(row=i+1, column=0, sticky="w", pady=2, padx=(2,6))
        ttk.Entry(stadler_frame, textvariable=var, width=18).grid(row=i+1, column=1, sticky="ew", padx=5, pady=2)

    # Ladanyi Parameters
    ladanyi_frame = ttk.LabelFrame(method_param_area, text="Ladanyi Parameters", padding=8)
    ladanyi_frame.grid_columnconfigure(1, weight=1)

    file_abs_coeff = tk.StringVar(value=r'C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\example_files\Multiple_spectra_manual\260122_azobenzene_40uM_MeOH_325nm_spectra_series_absorption_coefficients.csv')

    ttk.Label(ladanyi_frame, text="For Ladanyi: Absorption coefficient spectrum").grid(row=2, column=0, sticky="w", pady=(6,0))
    entry_frame2 = ttk.Frame(ladanyi_frame)
    entry_frame2.grid(row=3, column=0, sticky="ew", pady=2)
    entry_frame2.grid_columnconfigure(0, weight=1)
    ttk.Entry(entry_frame2, textvariable=file_abs_coeff).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(entry_frame2, text="Browse", command=lambda: choose_file(file_abs_coeff)).grid(row=0, column=1)

    # ========== RIGHT COLUMN ==========
    right_frame = ttk.Frame(root, padding=10)
    right_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.grid_rowconfigure(1, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)

    # Output text box
    output_label = ttk.Label(right_frame, text="Output", font=("TkDefaultFont", 10, "bold"))
    output_label.grid(row=0, column=0, sticky="w")
    
    output_box = tk.Text(right_frame, height=12, wrap="word")
    output_box.grid(row=1, column=0, sticky="nsew", pady=(5, 10))

    # Plot area
    plot_label = ttk.Label(right_frame, text="Plots", font=("TkDefaultFont", 10, "bold"))
    plot_label.grid(row=2, column=0, sticky="w")
    
    fig = plt.Figure(figsize=(12, 5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew", pady=(5, 0))

    def general_analyis():
        output_box.delete("1.0", tk.END)
        fig.clear()
        canvas.draw()

        def printer(x):
            output_box.insert(tk.END, x + "\n")
            output_box.see(tk.END)

        results_folder = results_folder_var.get()

        if not results_folder:
            messagebox.showerror(
                "Missing results folder",
                "Please select a folder where the results will be saved."
            )
            return None

        if not os.path.isdir(results_folder):
            messagebox.showerror(
                "Invalid results folder",
                "The selected results folder does not exist."
            )
            return None

        # Collect parameters
        params = {
            'Print function' : printer,
            "UV Vis file Multiple Spectra": file_UV_Vis_mult_spectra_var.get(),
            'UV Vis file start': file_UV_Vis_start.get(),
            "QYDS file": file_QYDS_var.get(),
            'Kinetics file': file_kinetics.get(),
            'Results Folder' : results_folder_var.get(),
            "Volume": float(volume_var.get()),
            'Concentration' : float(conc_var.get())*1E-6,
            "Irradiation wavelength": float(irradiation_var.get()),
            "Voltage after ref": float(voltage_var.get()),
            "Channel": channel_var.get(),
            "Measurement type": measurement_var.get(),
            "Automatic Measurements": manual_var.get(),
            "Color palette": palette_var.get(),
            "Plotting A_min": float(A_min_var.get()),
            "Plotting A_max": float(A_max_var.get()),
            "Plotting wl_min": float(wl_min_var.get()),
            "Plotting wl_max": float(wl_max_var.get()),
            "Legend fontsize spectra": int(font_legend_var.get()),
            "Plotting t_min": float(t_min_var.get()),
            "Plotting t_max": float(t_max_var.get()),
        }

        # If auto-time-spacing is used
        if manual_var.get() == "auto":
            params["Start"] = int(start_var.get())
            params["Stop"] = int(stop_var.get())
            params["Step"] = int(step_var.get())

        if params['Measurement type'] == 'Multiple Spectra':
            params['Basename'] = os.path.splitext(os.path.basename(params['UV Vis file Multiple Spectra']))[0]

        if params['Measurement type'] == 'Kinetics':
            params['Basename'] = 'Test'

        return params

    # Define the analysis functions
    def run_stadler():
        params = general_analyis()
        params["Kinetics wavelength"] = float(kinetics_var.get())
        params['Stadler QY'] = stadler_QY_var.get()
        params["Initial guess kinetics"] = [float(x) for x in initial_guess_var.get().split(",")]

        output_box.insert(tk.END, "Starting QY Calculation from Stadler\n")
        try:
            stadler_analysis(params=params)
            plotting_stadler(params, figure=fig)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))
 
    
    def run_ladanyi():
        params = general_analyis()
        params['Abs coeff file'] = file_abs_coeff.get()
        output_box.insert(tk.END, "Running Stanius analysis...\n")
        
        try:
            ladanyi_analysis(params=params)
            plotting_ladanyi(params, figure=fig)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e)) 

    start_button = ttk.Button(method_param_area, text="Start Stadler Analysis", command=run_stadler)
    start_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=5)

    # Tab change handler
    def on_tab_change(event):
        selected = notebook.index(notebook.select())

        # Hide all method-specific frames
        stadler_frame.grid_forget()
        ladanyi_frame.grid_forget()

        # Show the selected method frame
        if selected == 0:
            ladanyi_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            start_button = ttk.Button(method_param_area, text="Start Ladanyi Analysis", command=run_ladanyi)
            start_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=5)
        elif selected == 1:
            stadler_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            start_button = ttk.Button(method_param_area, text="Start Stadler Analysis", command=run_stadler)
            start_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=5)

    # Initialize with first tab
    ladanyi_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    
    notebook.bind("<<NotebookTabChanged>>", on_tab_change)

    root.mainloop()

if __name__ == "__main__":

    run_gui()