import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants
import sympy as sp
mpl.rcParams['mathtext.default'] = 'regular'
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

def produce_UV_Vis_dataframe(file):

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

def producte_QYDS_dataframe(file, 
                            rows_to_delete_indices_list=[], 
                            print_option=False):

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
    df = df.drop(rows_to_delete_indices_list)
    df = df.astype(float)

    if print_option:
        print(df)

    return df

def add_runtime_rows(df, start=0, stop=1200, step=9):
    
    df = df.iloc[0:0]
    """Append rows with runtime values and NaN for the other columns."""
    new_rows = pd.DataFrame({
        'Runtime': [pd.NA] * ((stop - start) // step + 1),
        'Illum Time': list(range(start, stop + 1, step)),
        'Cell Value': [pd.NA] * ((stop - start) // step + 1)
    })
    return pd.concat([df, new_rows], ignore_index=True)

def calculate_photon_flux(params, 
                          print_func=print):
    
    wavelength = params['Irradiation wavelength']
    voltage_after_reference = params['Voltage after ref']
    correction_factor = params['Conversion factor']
    volume = params['Volume']

    photon_flux_per_vol = wavelength * voltage_after_reference * correction_factor / (constants.Planck * constants.speed_of_light * constants.Avogadro * volume) * 1E-9
    photon_flux = photon_flux_per_vol * volume*1E-3

    print_func(r'Photon flux = {:.2e} mol s⁻¹ L⁻¹ / {:.2e} mol s⁻¹'.format(photon_flux_per_vol, photon_flux))

    params['Photon flux per vol'] = photon_flux_per_vol
    params['Photon flux'] = photon_flux

    return params

def extract_absorption_at_wavelength(params):

    # Extract parameters
    dataframe = params['UV Vis df']
    wavelength = params['Kinetics wavelength']

    # Find the row index closest to the specified wavelength
    closest_row_index = (dataframe.iloc[:, 0] - wavelength).abs().idxmin()
    
    # Extract absorption values at the closest row index
    absorption_values = dataframe.iloc[closest_row_index, 1:].tolist()

    params['Kinetics abs'] = absorption_values

    return params

def fit_absorbance_vs_time_first_order(params, print_func=print):

    initial_guess = params['Initial guess kinetics']        # [A_inf, A_i, k]
    absorbances = np.asarray(params['Kinetics abs'], dtype=float)
    df_QYDS = params['QYDS df']
    time = np.asarray(df_QYDS.iloc[:, 1], dtype=float)

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
    A_min = params['Plotting A_min']
    A_max = params['Plotting A_max']
    wl_min = params['Plotting wl_min']
    wl_max = params['Plotting wl_max']
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
    ax.set_xlim(wl_min, wl_max)
    ax.set_ylim(A_min, A_max)
    ax.legend(fontsize=fontsize_legend)

    return ax

def plotting_photokinetics(params, ax=None):

    absorbances = params['Kinetics abs']
    df_QYDS = params['QYDS df']
    time = df_QYDS.iloc[:, 1]
    time = np.array(time)
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
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.legend()

    return ax

def plotting(params, figure=None, testing=True):
    if figure is None:
        fig = plt.figure(layout='constrained', figsize=(14, 6))
    else:
        fig = figure
        fig.clf()  # Clear previous plots

    subfig = fig.subfigures(1, 1)
    axs = subfig.subplots(1, 2, sharey=True)

    plotting_spectra(params, ax=axs[0])
    plotting_photokinetics(params, ax=axs[1])

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

def calculate_QY(params,
                 print_func=print):

    k_fit = params['k']
    c_0 = params['Concentration']
    photon_flux = params['Photon flux per vol']
    A_i_0 = get_initial_absorbance_at_irradiation_wavelength(params=params)
    I_abs = photon_flux*(1-10**(-A_i_0))
    print(f'Ai0:{A_i_0}, photon flux per vol: {photon_flux}, I_abs : {I_abs}')

    QY = k_fit * c_0 / (photon_flux * (1 - 10**(-A_i_0)))
    
    print_func('quantum yield = {:.2f}'.format(QY) )

def calculate_pseudo_total_QY(params,
                              print_func=print):
    
    k_fit = params['k']
    photon_flux = params['Photon flux per vol']
    A_i_0 = get_initial_absorbance_at_irradiation_wavelength(params=params)

    photo_kinetic_factor = ( 1 - 10**(-A_i_0) ) / A_i_0
    Q = k_fit / ( 1 * photon_flux * photo_kinetic_factor ) 

    print_func('pseudo total QY = {:.2e} L mol⁻¹ cm⁻¹'.format(Q))

def get_testing_parameters(scenario=1):

    if scenario == 1:

        # Get Files
        path = os.getcwd()
        file_UV_Vis = os.path.join(path, r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251127_AzoAlaNH2_cycling_365_455_645nm.csv")
        file_QYDS = os.path.join(path, r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251117_C1_365nm.dat")

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 365 # nm
        conversion_factor = 0.2315 # mW / V
        voltage_after_ref = 3.29 # V
        kinetics_wavelength = 350 # nm
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
        path = os.getcwd()
        file_UV_Vis = os.path.join(path, r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\240409.csv")
        file_QYDS = os.path.join(path, r"testfiles\240409_CM27.dat")

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 18E-6 # M
        irradiation_wavelength = 365 # nm
        conversion_factor = 0.25 # mW / V
        voltage_after_ref = 5.80 # V
        kinetics_wavelength = 350 # nm
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
        path = os.getcwd()
        file_UV_Vis = os.path.join(path, r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\oNBA_Water_365nm_100mA.csv")
        file_QYDS = os.path.join(path, r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\oNBA_Water_365nm_100mA.dat")

        # Measurement parameters
        volume = 2 # mL
        conc_initial = 54.5E-6 # M
        irradiation_wavelength = 365 # nm
        conversion_factor = 4.9 # mW / V
        # voltage_after_ref = 7.49 # V
        voltage_after_ref = 2.21 # V
        kinetics_wavelength = 280 # nm
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
        path = os.getcwd()
        file_UV_Vis = r"testfiles\251204_305nm.csv"
        file_QYDS =  r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251204_2nd_305nm.dat"

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 305 # nm
        conversion_factor = 0.399 # mW / V
        voltage_after_ref = 7.12 # V
        kinetics_wavelength = 540 # nm
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
        path = os.getcwd()
        file_UV_Vis = r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251204_521nm.csv"
        file_QYDS =  r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\testfiles\251204_2nd_305_PSS_521nm.dat"

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 521 # nm
        conversion_factor = 0.130 # mW / V
        voltage_after_ref = 9.31 # V
        kinetics_wavelength = 540 # nm
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

    return {
        'UV Vis file' : file_UV_Vis,
        'QYDS file' : file_QYDS,
        'Volume' : volume,
        'Concentration' : conc_initial,
        'Irradiation wavelength' : irradiation_wavelength,
        'Conversion factor' : conversion_factor,
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
    }

def testing_main(scenario=1):

    # Get Parameters
    params = get_testing_parameters(scenario=scenario)

    
    # Create Dataframes
    df_UV_Vis = produce_UV_Vis_dataframe(file=params['UV Vis file'])
    params['UV Vis df'] = df_UV_Vis
    df_QYDS = producte_QYDS_dataframe(file=params['QYDS file'],
                                      rows_to_delete_indices_list=[],
                                      print_option=False)
    if scenario == 3:
        df_QYDS = add_runtime_rows(df=df_QYDS,
                            start=0,
                            stop=162,
                            step=9)
    params['QYDS df'] = df_QYDS

    if len(df_QYDS) != len(df_UV_Vis.columns) - 1:
        print(f'QYDS dataframe ({len(df_QYDS)}) and irradiation times ({len(df_UV_Vis.columns)}) have not the same length!')
        return
    
    params = extract_absorption_at_wavelength(params=params)
    
    params = calculate_photon_flux(params=params,
                                   print_func=print)

    params = fit_absorbance_vs_time_first_order(params,
                                                print_func=print)

    plotting(params=params, testing=True)

    calculate_QY(params=params,
                 print_func=print)

    calculate_pseudo_total_QY(params=params,
                              print_func=print)

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
    
    # Make the window open in full screen
    root.state('zoomed')  # For Windows
    # Alternatively for cross-platform: root.attributes('-zoomed', True)
    
    # Configure grid weights for proper resizing
    root.grid_columnconfigure(0, weight=0)  # Left column fixed
    root.grid_columnconfigure(1, weight=1)  # Right column expands
    root.grid_rowconfigure(0, weight=1)     # Allow vertical expansion

    # ------------------- LEFT COLUMN -------------------
    left_frame = ttk.Frame(root, padding=10)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
    
    # ------------------- File selection (TOP) -------------------
    file_frame = ttk.LabelFrame(left_frame, text="File Selection", padding=10)
    file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

    def choose_file(var):
        file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        var.set(file)

    file_UV_Vis_var = tk.StringVar()
    file_QYDS_var = tk.StringVar()

    ttk.Label(file_frame, text="UV Vis file:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(file_frame, textvariable=file_UV_Vis_var, width=40).grid(row=1, column=0, padx=5, pady=2)
    ttk.Button(file_frame, text="Browse", command=lambda: choose_file(file_UV_Vis_var)).grid(row=2, column=0, padx=5, pady=2, sticky="ew")

    ttk.Label(file_frame, text="QYDS file:").grid(row=3, column=0, sticky="w", pady=(10, 2))
    ttk.Entry(file_frame, textvariable=file_QYDS_var, width=40).grid(row=4, column=0, padx=5, pady=2)
    ttk.Button(file_frame, text="Browse", command=lambda: choose_file(file_QYDS_var)).grid(row=5, column=0, padx=5, pady=2, sticky="ew")

    # ------------------- Measurement parameters (MIDDLE) -------------------
    meas_frame = ttk.LabelFrame(left_frame, text="Measurement Parameters", padding=10)
    meas_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

    volume_var = tk.StringVar(value='3')
    conc_var = tk.StringVar(value='50')
    irradiation_var = tk.StringVar(value='365')
    conversion_var = tk.StringVar(value='0.2315')
    voltage_var = tk.StringVar(value='3.29')
    kinetics_var = tk.StringVar(value='350')
    initial_guess_var = tk.StringVar(value='0,0,0')

    meas_params = [
        ("Volume", volume_var),
        ("Concentration", conc_var),
        ("Irradiation wavelength", irradiation_var),
        ("Conversion factor", conversion_var),
        ("Voltage after reference cuvette", voltage_var),
        ("Kinetics wavelength", kinetics_var),
        ("Initial guess kinetics", initial_guess_var)
    ]

    for i, (label, var) in enumerate(meas_params):
        ttk.Label(meas_frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=2)
        ttk.Entry(meas_frame, textvariable=var, width=20).grid(row=i, column=1, pady=2, padx=5)

    # ------------------- Plotting parameters (BOTTOM) -------------------
    plot_frame = ttk.LabelFrame(left_frame, text="Plotting Parameters", padding=10)
    plot_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))

    palette_var = tk.StringVar(value='crest')
    A_min_var = tk.StringVar(value='0')
    A_max_var = tk.StringVar(value='0.8')
    wl_min_var = tk.StringVar(value='300')
    wl_max_var = tk.StringVar(value='800')
    font_legend_var = tk.StringVar(value='10')
    t_min_var = tk.StringVar(value='-10')
    t_max_var = tk.StringVar(value='610')

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

    # ------------------- Run button -------------------
    ttk.Button(left_frame, text="Run Analysis", command=lambda: run_analysis()).grid(row=3, column=0, pady=10, sticky="ew")

    # ------------------- RIGHT COLUMN -------------------
    right_frame = ttk.Frame(root, padding=10)
    right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    right_frame.grid_rowconfigure(0, weight=0)  # Plot area - fixed height
    right_frame.grid_rowconfigure(1, weight=1)  # Output box - expandable
    right_frame.grid_columnconfigure(0, weight=1)
    
    # Plot display
    plot_display_frame = ttk.LabelFrame(right_frame, text="Plot Display", padding=10)
    plot_display_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

    # Create an empty figure as placeholder
    fig = plt.Figure(figsize=(14, 6), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_display_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=False)
    
    # Output box
    output_frame = ttk.LabelFrame(right_frame, text="Output", padding=10)
    output_frame.grid(row=1, column=0, sticky="nsew")
    
    # Create text widget with scrollbar
    output_scrollbar = ttk.Scrollbar(output_frame)
    output_scrollbar.pack(side='right', fill='y')
    
    output_text = tk.Text(output_frame, height=10, wrap='word', yscrollcommand=output_scrollbar.set)
    output_text.pack(side='left', fill='both', expand=True)
    output_scrollbar.config(command=output_text.yview)
    
    # Function to print to output box
    def print_to_output(message):
        output_text.insert('end', message + '\n')
        output_text.see('end')  # Auto-scroll to bottom

    # ------------------- Run analysis function -------------------
    def run_analysis():
        output_text.delete('1.0', tk.END)
        initial_guess_str = initial_guess_var.get()
        initial_guess_list = [float(x.strip()) for x in initial_guess_str.split(',')]
        concentation = float(conc_var.get())
        concentation = concentation * 1E-6
        params = {
            'UV Vis file': file_UV_Vis_var.get(),
            'QYDS file': file_QYDS_var.get(),
            'Volume': float(volume_var.get()),
            'Concentration': concentation,
            'Irradiation wavelength': float(irradiation_var.get()),
            'Conversion factor': float(conversion_var.get()),
            'Voltage after ref': float(voltage_var.get()),
            'Kinetics wavelength': float(kinetics_var.get()),
            'Initial guess kinetics': initial_guess_list,
            'Color palette': palette_var.get(),
            'Plotting A_min': float(A_min_var.get()),
            'Plotting A_max': float(A_max_var.get()),
            'Plotting wl_min': float(wl_min_var.get()),
            'Plotting wl_max': float(wl_max_var.get()),
            'Legend fontsize spectra': float(font_legend_var.get()),
            'Plotting t_min': float(t_min_var.get()),
            'Plotting t_max': float(t_max_var.get()),
        }
        # Data processing
        df_UV_Vis = produce_UV_Vis_dataframe(file=params['UV Vis file'])
        params['UV Vis df'] = df_UV_Vis
        df_QYDS = producte_QYDS_dataframe(file=params['QYDS file'],
                                        rows_to_delete_indices_list=[],
                                        print_option=False)
        params['QYDS df'] = df_QYDS
        if len(df_QYDS) != len(df_UV_Vis.columns) - 1:
            print(f'QYDS dataframe ({len(df_QYDS)}) and irradiation times ({len(df_UV_Vis.columns)}) have not the same length!')
            return

        params = extract_absorption_at_wavelength(params=params)
        
        params = calculate_photon_flux(params=params,
                                       print_func=print_to_output)

        params = fit_absorbance_vs_time_first_order(params,
                                                    print_func=print_to_output)

        # Update the plot
        plotting(params=params, figure=fig)
        canvas.draw()

        calculate_QY(params=params,
                     print_func=print_to_output)

        calculate_pseudo_total_QY(params=params,
                                  print_func=print_to_output)

    root.mainloop()

def main(testing=False,
         testing_scenario=1):
    
    if testing:
        testing_main(scenario=testing_scenario)
    else:
        run_gui()

if __name__ == "__main__":

    main(testing=True,
         testing_scenario=4)