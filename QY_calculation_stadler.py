import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants
# mpl.rcParams['mathtext.default'] = 'regular'
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

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

def produce_UV_Vis_dataframe(params):

    file = params['UV Vis file']

    df = read_in_UV_Vis_csv(file=file)

    params['UV Vis df'] = df

    return params

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

    photon_flux_per_vol = wavelength * voltage_after_reference * correction_factor / (constants.Planck * constants.speed_of_light * constants.Avogadro * volume) * 1E-9
    photon_flux = photon_flux_per_vol * volume*1E-3

    print_func(f'Conversion factor = {correction_factor:.2f} mW V⁻¹')
    print_func(r'Photon flux = {:.2e} mol s⁻¹ L⁻¹ / {:.2e} mol s⁻¹'.format(photon_flux_per_vol, photon_flux))

    params['Photon flux per vol'] = photon_flux_per_vol
    params['Photon flux'] = photon_flux

    return params

def calculate_P_abs(params):

    print_func = params['Print function']
    ref = params['Voltage after ref']
    sample = params['Voltage after sample']
    correction_factor = get_correction_factor(params=params)

    P_ref = ref * correction_factor * 1E-3
    P_sample = sample * correction_factor * 1E-3
    R = 0.036

    f = (1 + R*(P_sample/P_ref)) / (1-R)

    P_abs = (ref - sample)*f * 1E-3
    print_func(f'P abs = {P_abs:.4e} W')

    params['P abs'] = P_abs

    return params


def create_kinetics_df(params):

    if params['Measurement type'] == 'Multiple Spectra':

        # Extract parameters
        dataframe = params['UV Vis df']
        wavelength = params['Kinetics wavelength']
        df_QYDS = params['QYDS df']
        time = np.asarray(df_QYDS.iloc[:, 1], dtype=float)

        # Find the row index closest to the specified wavelength
        closest_row_index = (dataframe.iloc[:, 0] - wavelength).abs().idxmin()
        
        # Extract absorption values at the closest row index
        absorption_values = dataframe.iloc[closest_row_index, 1:].tolist()

        df_kinetics = pd.DataFrame({
            'Time / s' : time,
            'Absorbance' : absorption_values
        })

        params['Kinetics df'] = df_kinetics

        return params
    
    if params['Measurement type'] == 'Kinetics':
        
        file = params['Kinetics file']

        df = read_in_UV_Vis_csv(file=file)

        df_kinetics = pd.DataFrame({
            'Time / s' : df.iloc[:,0]*60,
            'Absorbance' : df.iloc[:,1]
        })

        params['Kinetics df'] = df_kinetics

def fit_absorbance_vs_time_first_order(params):

    print_func = params['Print function']
    initial_guess = params['Initial guess kinetics']        # [A_inf, A_i, k]
    df_kinetics = params['Kinetics df']
    absorbances = df_kinetics['Absorbance'].to_numpy()
    time = df_kinetics['Time / s'].to_numpy()

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

    df_QYDS = params['QYDS df']
    irradiation_times = df_QYDS.iloc[:, 1]
    df_UV_Vis = params['UV Vis df']
    palette_name = params['Color palette']
    ymin = params['Plotting A_min']
    ymax = params['Plotting A_max']
    xmin = params['Plotting wl_min']
    xmax = params['Plotting wl_max']
    fontsize_legend = params['Legend fontsize spectra']

    # Create own figure if subplots are OFF
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))

    palette = sns.color_palette(palette_name, n_colors=len(irradiation_times))

    for i, col in enumerate(df_UV_Vis.columns[1:], start=1):
        x = df_UV_Vis.iloc[:, 0]
        y = df_UV_Vis.iloc[:, i]
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
    absorbances = df_kinetics['Absorbance'].to_numpy()
    time = df_kinetics['Time / s'].to_numpy()
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

def plotting(params, figure=None):

    testing = params['Testing']
    
    if figure is None:
        fig = plt.figure(layout='constrained', figsize=(12, 5))
    else:
        fig = figure
        fig.clf()  # Clear previous plots

    axs = fig.subplots(1, 2, sharey=True)
    if params['Measurement type'] == 'Multiple Spectra':
        plotting_spectra(params, ax=axs[0])
        plotting_photokinetics(params, ax=axs[1])
    else:
        plotting_photokinetics(params, ax=axs[0])

    if testing:
        plt.show()

    return fig

def get_initial_absorbance_at_irradiation_wavelength(params):

    dataframe = params['UV Vis df']
    wavelength = params['Irradiation wavelength']

    # Find the row index closest to the specified wavelength
    closest_row_index = (dataframe.iloc[:, 0] - wavelength).abs().idxmin()
    
    # Extract absorption values at the closest row index
    absorption_values = dataframe.iloc[closest_row_index, 1:].tolist()

    abs_i0 = absorption_values[0]

    return abs_i0

def calculate_QY(params):

    print_func = params['Print function']
    k_fit = params['k']
    c_0 = params['Concentration']
    photon_flux = params['Photon flux per vol']
    A_i_0 = get_initial_absorbance_at_irradiation_wavelength(params=params)
    I_abs = photon_flux*(1-10**(-A_i_0))

    QY_stadler = k_fit * c_0 / (photon_flux * (1 - 10**(-A_i_0)))
    
    print_func(f'A(i,0) = {A_i_0:.2f}')
    print_func('quantum yield from Stadler = {:.2f}'.format(QY_stadler) )

    V = params['Volume'] 
    P_abs = params['P abs']
    t = params['Irradiation time']
    wl = params["Irradiation wavelength"]

    QY_QYDS = (
        constants.Avogadro * constants.speed_of_light * constants.Planck *
        ( 30.66 * V  ) / 
        ( P_abs * t *  wl )
    )  
    
    print_func(f'quantum yield from Megerle = {QY_QYDS}')



def calculate_pseudo_total_QY(params):

    print_func = params['Print function']
    k_fit = params['k']
    photon_flux = params['Photon flux per vol']
    A_i_0 = get_initial_absorbance_at_irradiation_wavelength(params=params)

    photo_kinetic_factor = ( 1 - 10**(-A_i_0) ) / A_i_0
    Q = k_fit / ( 1 * photon_flux * photo_kinetic_factor ) 

    print_func('pseudo total QY = {:.2e} L mol⁻¹ cm⁻¹'.format(Q))

def get_testing_parameters(scenario=1):

    if scenario == 1:

        # Get Files
        file_UV_Vis = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251127_AzoAlaNH2_cycling_365_455_645nm.csv"
        file_QYDS = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251117_C1_365nm.dat"
        file_kinetics = r''

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 365 # nm
        channel = 'low'
        voltage_after_ref = 3.29 # V
        kinetics_wavelength = 350 # nm
        measurement = "Multiple Spectra"
        manual = 'manual'
        start = 0
        stop = 0
        step = 0
        kinetics_p0 = [0,0,0]

        # Plotting parameters
        palette = 'crest'
        A_min = 0
        A_max = 0.8
        wl_min = 300
        wl_max = 800
        font_legend_spectra = 10
        t_min = -10
        t_max = 610

    if scenario == 2:

        # Get Files
        file_UV_Vis = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\240409.csv"
        file_QYDS =  r"testfiles\240409_CM27.dat"
        file_kinetics = r''

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 18E-6 # M
        irradiation_wavelength = 365 # nm
        channel = 'low'
        voltage_after_ref = 5.80 # V
        kinetics_wavelength = 350 # nm
        measurement = "Multiple Spectra"
        manual = 'manual'
        start = 0
        stop = 0
        step = 0
        kinetics_p0 = [0,0,0]

        # Plotting parameters
        palette = 'crest'
        A_min = 0
        A_max = 0.25
        wl_min = 300
        wl_max = 600
        font_legend_spectra = 8
        t_min = -10
        t_max = 610

    if scenario == 3:
    
        # Get Files
        file_UV_Vis = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\oNBA_Water_365nm_100mA.csv"
        file_QYDS = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\oNBA_Water_365nm_100mA.dat"
        file_kinetics = r''

        # Measurement parameters
        volume = 2 # mL
        conc_initial = 54.5E-6 # M
        irradiation_wavelength = 365 # nm
        channel = 'high'
        # voltage_after_ref = 7.49 # V
        voltage_after_ref = 2.21 # V
        kinetics_wavelength = 280 # nm
        measurement = "Multiple Spectra"
        manual = 'auto'
        start = 0
        stop = 162
        step = 9
        kinetics_p0 = []

        # Plotting parameters
        palette = 'crest'
        A_min = 0
        A_max = 1
        wl_min = 200
        wl_max = 500
        font_legend_spectra = 8
        t_min = -5
        t_max = 175

    if scenario == 4:

        # Get Files
        file_UV_Vis = r"testfiles\251204_305nm.csv"
        file_QYDS =  r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251204_2nd_305nm.dat"
        file_kinetics = r''

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 305 # nm
        channel = 'low'
        voltage_after_ref = 7.12 # V
        kinetics_wavelength = 540 # nm
        measurement = "Multiple Spectra"
        manual = 'manual'
        start = 0
        stop = 0
        step = 0
        kinetics_p0 = []

        # Plotting parameters
        palette = 'viridis'
        A_min = 0
        A_max = 0.8
        wl_min = 250
        wl_max = 800
        font_legend_spectra = 10
        t_min = -5
        t_max = 500
    
    if scenario == 5:

        # Get Files
        file_UV_Vis = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251204_521nm.csv"
        file_QYDS =  r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251204_2nd_305_PSS_521nm.dat"
        file_kinetics = r''

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 521 # nm
        channel = 'low'
        voltage_after_ref = 9.31 # V
        kinetics_wavelength = 540 # nm
        measurement = "Multiple Spectra"
        manual = 'manual'
        start = 0
        stop = 0
        step = 0
        kinetics_p0 = []

        # Plotting parameters
        palette = 'viridis'
        A_min = -0.01
        A_max = 0.8
        wl_min = 250
        wl_max = 800
        font_legend_spectra = 10
        t_min = -5
        t_max = 605

    if scenario == 6:

        # Get Files
        file_UV_Vis = r"E:\01_Raw\01_Azobenzene\UV_Vis\251205\251205_HAB_42uM_dioxane_teq.csv"
        file_QYDS =  r""
        file_kinetics = r"E:\01_Raw\01_Azobenzene\UV_Vis\251205\251205_HAB_dioxane_kinetics365nm.csv"

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 42E-6 # M
        irradiation_wavelength = 365 # nm
        channel = 'low'
        voltage_after_ref = 3.46 # V
        kinetics_wavelength = 365 # nm
        measurement = "Kinetics"
        manual = 'manual'
        start = 0
        stop = 0
        step = 0
        kinetics_p0 = []

        # Plotting parameters
        palette = 'viridis'
        A_min = 0.8
        A_max = 1.7
        wl_min = 250
        wl_max = 800
        font_legend_spectra = 10
        t_min = -5
        t_max = 550

    return {
        'Testing' : True,
        'Print function' : print,
        'UV Vis file' : file_UV_Vis,
        'QYDS file' : file_QYDS,
        'Kinetics file' : file_kinetics,
        'Volume' : volume,
        'Concentration' : conc_initial,
        'Irradiation wavelength' : irradiation_wavelength,
        'Measurement type' : measurement,
        'Channel' : channel,
        'Automatic Measurements' : manual,
        'Voltage after ref' : voltage_after_ref,
        'Kinetics wavelength' : kinetics_wavelength,
        'Initial guess kinetics' : kinetics_p0,
        'Color palette' : palette,
        'Plotting A_min' : A_min,
        'Plotting A_max' : A_max,
        'Plotting wl_min' : wl_min,
        'Plotting wl_max' : wl_max,
        'Legend fontsize spectra' : font_legend_spectra,
        'Plotting t_min' : t_min,
        'Plotting t_max' : t_max,
        'Start' : start,
        'Stop' : stop,
        'Step' : step,
    }

def analyze(params=None,
            scenario=1):
    
    if not params:
        params = get_testing_parameters(scenario=scenario)
    
    if params['Measurement type'] == 'Multiple Spectra':
        produce_UV_Vis_dataframe(params=params)
        df_UV_Vis = params['UV Vis df']
        producte_QYDS_dataframe(params=params)
        df_QYDS = params['QYDS df']
        if len(df_QYDS) != len(df_UV_Vis.columns) - 1:
            print(f'QYDS dataframe ({len(df_QYDS)}) and irradiation times ({len(df_UV_Vis.columns)}) have not the same length!')
            return
        
        create_kinetics_df(params=params)

    if params['Measurement type'] == 'Kinetics':
        produce_UV_Vis_dataframe(params=params)
        create_kinetics_df(params=params)

    calculate_photon_flux(params=params)
    calculate_P_abs(params=params)
    fit_absorbance_vs_time_first_order(params=params)
    calculate_QY(params=params)
    calculate_pseudo_total_QY(params=params)
    if params['Testing']:
        plotting(params=params)
        return
    if not params['Testing']:
        return params

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
    
    # Configure grid weights for proper resizing
    root.grid_columnconfigure(0, weight=0)  # Left column fixed
    root.grid_columnconfigure(1, weight=1)  # Right column expands
    root.grid_rowconfigure(0, weight=1)     # Allow vertical expansion

    # Left column
    left_frame = ttk.Frame(root, padding=10)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
    
    # File selection
    file_frame = ttk.LabelFrame(left_frame, text="File Selection", padding=10)
    file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

    def choose_file(var):
        file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        var.set(file)

    file_UV_Vis_var = tk.StringVar(value=r"E:\01_Raw\01_Azobenzene\UV_Vis\251205\251205_HAB_42uM_dioxane_teq.csv")
    file_QYDS_var = tk.StringVar(value=r'')
    file_initial_spectrum_var = tk.StringVar(value=r"E:\01_Raw\01_Azobenzene\UV_Vis\251205\251205_HAB_dioxane_kinetics365nm.csv")

    ttk.Label(file_frame, text="UV Vis file:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(file_frame, textvariable=file_UV_Vis_var, width=40).grid(row=1, column=0, padx=5, pady=2)
    ttk.Button(file_frame, text="Browse", command=lambda: choose_file(file_UV_Vis_var)).grid(row=2, column=0, padx=5, pady=2, sticky="ew")

    ttk.Label(file_frame, text="QYDS file:").grid(row=3, column=0, sticky="w", pady=(10, 2))
    ttk.Entry(file_frame, textvariable=file_QYDS_var, width=40).grid(row=4, column=0, padx=5, pady=2)
    ttk.Button(file_frame, text="Browse", command=lambda: choose_file(file_QYDS_var)).grid(row=5, column=0, padx=5, pady=2, sticky="ew")

    ttk.Label(file_frame, text="In case of Kinetics: Kinetic file:").grid(row=6, column=0, sticky="w", pady=(10, 2))
    ttk.Entry(file_frame, textvariable=file_initial_spectrum_var, width=40).grid(row=7, column=0, padx=5, pady=2)
    ttk.Button(file_frame, text="Browse", command=lambda: choose_file(file_initial_spectrum_var)).grid(row=8, column=0, padx=5, pady=2, sticky="ew")

    # General Parameteers
    gen_frame = ttk.LabelFrame(left_frame, text="General Parameters", padding=10)
    gen_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))

    ttk.Label(gen_frame, text="Channel:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    channel_var = tk.StringVar()
    ttk.Combobox(
        gen_frame,
        textvariable=channel_var,
        values=["high", "low"],
        state="readonly"
    ).grid(row=0, column=1, padx=5, pady=2)
    channel_var.set("low")  

    ttk.Label(gen_frame, text="Measurement:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    measurement_var = tk.StringVar()
    ttk.Combobox(
        gen_frame,
        textvariable=measurement_var,
        values=["Multiple Spectra", "Kinetics"],
        state="readonly"
    ).grid(row=1, column=1, padx=5, pady=2)
    measurement_var.set("Kinetics")  
    
    ttk.Label(gen_frame, text="Manual/Auto:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    manual_var = tk.StringVar()
    ttk.Combobox(
        gen_frame,
        textvariable=manual_var,
        values=["manual", "auto"],
        state="readonly"
    ).grid(row=2, column=1, padx=5, pady=2)
    manual_var.set("manual")  

    # Measurement parameters
    meas_frame = ttk.LabelFrame(left_frame, text="Measurement Parameters", padding=10)
    meas_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

    volume_var = tk.StringVar(value='3')
    conc_var = tk.StringVar(value='42')
    irradiation_var = tk.StringVar(value='365')
    voltage_var = tk.StringVar(value='3.46')
    voltage_sample_var = tk.StringVar(value='3.287')
    irradiation_time_var = tk.StringVar(value='545')
    kinetics_var = tk.StringVar(value='350')
    initial_guess_var = tk.StringVar(value='0,0,0')
    start_var = tk.StringVar(value='0')
    stop_var = tk.StringVar(value='0')
    step_var = tk.StringVar(value='0')

    meas_params = [
        ("Volume", volume_var),
        ("Concentration", conc_var),
        ("Irradiation wavelength", irradiation_var),
        ("Voltage after reference cuvette", voltage_var),
        ('Voltage after sample', voltage_sample_var),
        ('Irradiation time', irradiation_time_var),
        ("Kinetics wavelength", kinetics_var),
        ("Initial guess kinetics", initial_guess_var),
        ('Start', start_var),
        ('Stop', stop_var),
        ('Step', step_var)
    ]

    for i, (label, var) in enumerate(meas_params):
        ttk.Label(meas_frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=2)

        if isinstance(var, tk.BooleanVar):
            ttk.Checkbutton(meas_frame, variable=var).grid(row=i, column=1, sticky="w", padx=5, pady=2)
        else:
            ttk.Entry(meas_frame, textvariable=var, width=20).grid(row=i, column=1, pady=2, padx=5)

    # Plotting parameters
    plot_frame = ttk.LabelFrame(left_frame, text="Plotting Parameters", padding=10)
    plot_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 10))

    palette_var = tk.StringVar(value='crest')
    A_min_var = tk.StringVar(value='0')
    A_max_var = tk.StringVar(value='0')
    wl_min_var = tk.StringVar(value='0')
    wl_max_var = tk.StringVar(value='0')
    font_legend_var = tk.StringVar(value='0')
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
        ("t_max", t_max_var)
    ]

    for i, (label, var) in enumerate(plot_params):
        ttk.Label(plot_frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=2)
        ttk.Entry(plot_frame, textvariable=var, width=20).grid(row=i, column=1, pady=2, padx=5)

    # --------------------------
    # OUTPUT FRAME (RIGHT SIDE)
    # --------------------------
    right_frame = ttk.Frame(root, padding=10)
    right_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.grid_rowconfigure(1, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)

    # Button to start analysis
    run_button = ttk.Button(right_frame, text="Run Analysis", command=lambda: run_analysis())
    run_button.grid(row=0, column=0, sticky="ew", pady=5)

    # Text output window
    output_box = tk.Text(right_frame, height=12, wrap="word")
    output_box.grid(row=1, column=0, sticky="nsew", pady=5)

    # --------------------------
    # MATPLOTLIB CANVAS
    # --------------------------
    fig = plt.Figure(figsize=(12, 5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", pady=5)

    # --------------------------
    # MAIN RUN FUNCTION
    # --------------------------
    def run_analysis():
        output_box.delete("1.0", tk.END)

        def printer(x):
            output_box.insert(tk.END, x + "\n")
            output_box.see(tk.END)

        # Collect parameters
        params = {
            'Testing' : False,
            'Print function' : printer,
            "UV Vis file": file_UV_Vis_var.get(),
            "QYDS file": file_QYDS_var.get(),
            'Kinetics file': file_initial_spectrum_var.get(),
            "Volume": float(volume_var.get()),
            "Concentration": float(conc_var.get()) * 1e-6,  # µM → M?
            "Irradiation wavelength": float(irradiation_var.get()),
            "Voltage after ref": float(voltage_var.get()),
            'Voltage after sample' : float(voltage_sample_var.get()),
            'Irradiation time' : float(irradiation_time_var.get()),
            "Kinetics wavelength": float(kinetics_var.get()),
            "Initial guess kinetics": [float(x) for x in initial_guess_var.get().split(",")],
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
            params["Start"] = float(start_var.get())
            params["Stop"] = float(stop_var.get())
            params["Step"] = float(step_var.get())

        try:
            analyze(params=params)
            # Plot
            fig.clf()
            plotting(params, figure=fig)
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root.mainloop()

def main(testing=False,
         testing_scenario=1):
    
    if testing:
        analyze(scenario=testing_scenario)
    else:
        run_gui()

if __name__ == "__main__":

    main(testing=False,
         testing_scenario=6)