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

def produce_UV_Vis_dataframe(file, 
                             last_row_with_data, 
                             first_column_with_data, 
                             last_column_with_data, 
                             print_option=False):

    df = pd.read_csv(file)

    # delete columns and rows
    df = df.iloc[1:last_row_with_data-1, first_column_with_data:last_column_with_data]

    # Shift the column names one to the right
    new_column_names = [None] + list(df.columns[:-1])
    df.columns = new_column_names

    # Delete all wavelenth column except first
    columns_to_delete = [i for i in range(1, len(df.columns)) if i % 2 == 0]
    df = df.drop(columns=df.columns[columns_to_delete])
    df = df.astype(float)

    if print_option:
        print(df)

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

def calculate_photon_flux(params):
    
    wavelength = params['Irradiation wavelength']
    voltage_after_reference = params['Voltage after ref']
    correction_factor = params['Conversion factor']
    volume = params['Volume']

    photon_flux = wavelength * voltage_after_reference * correction_factor / (constants.Planck * constants.speed_of_light * constants.Avogadro * volume) * 1E-12

    print(r'Photon flux = {:.2e} mol s⁻¹ L⁻¹'.format(photon_flux))

    return photon_flux

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



def plot_photokinetics(irradiation_times, palette, dataframe, xmin, xmax, ymin, ymax, fontsize_legend):

    plt.rcParams.update({'font.size': 12})
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Create a color palette based on the concentrations list
    palette = sns.color_palette(palette, n_colors=len(irradiation_times))

    # Determine the minimum and maximum concentration values
    min_time = min(irradiation_times)
    max_time = max(irradiation_times)

    for i, column in enumerate(dataframe.columns[1:], start=1):
        x_values = dataframe.iloc[:, 0]  # X-values from the first column
        y_values = dataframe.iloc[:, i]  # Y-values from each subsequent column
        time = irradiation_times[i - 1]  # Get concentration corresponding to the column

        # Scale the concentration value to fit within the range of available colors
        scaled_time = (time - min_time) / (max_time - min_time)
        color_index = int(scaled_time * (len(palette) - 1))  # Scale to fit within palette range
        color = palette[color_index]  # Choose color based on scaled concentration value

        plt.plot(x_values, y_values, label="{:.0f} s".format(time), color=color)

    plt.xlabel("wavelength / nm")  # Update with your x-axis label
    plt.ylabel("absorbance")  # Update with your y-axis label
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.legend(fontsize=fontsize_legend)
    plt.show()

def fit_absorbance_vs_time_first_order(params):

    # Extract Parameters
    initial_guess = params['Initial guess kinetics']
    absorbances = params['Kinetics abs']
    df_QYDS = params['QYDS df']
    time = df_QYDS.iloc[:,1]

    time = np.array(time)

    # Define fitting function
    def fitting_function(t, A_inf, A_i, k):
        return A_inf +  (A_i - A_inf) * np.exp(-k * t)

    # Perform curve fit
    fit_parameters, covariance = curve_fit(fitting_function, time, absorbances, p0=initial_guess)

    # Calculate standard deviations (errors) of fit parameters from covariance matrix
    fit_parameter_errors = np.sqrt(np.diag(covariance))

    if print_option == 'yes':
        # Print fit parameters with errors
        print('A_inf = {:.5f} ± {:.5f}'.format(fit_parameters[0], fit_parameter_errors[0]))
        print('A_i = {:.5f} ± {:.5f}'.format(fit_parameters[1], fit_parameter_errors[1]))
        print('k = {:.5f} ± {:.5f}'.format(fit_parameters[2], fit_parameter_errors[2]))

    return fit_parameters[0], fit_parameters[1], fit_parameters[2], fit_parameter_errors[2]

def plot_absorbance_vs_time_with_fit_first_order(initial_guess, absorbances, time, xmin, xmax, ymin, ymax, wavelength):

    plt.rcParams.update({'font.size': 12})

    # Define fitting function
    def fitting_function(t, A_inf, A_i, k):
        return A_inf +  (A_i - A_inf) * np.exp(-k * t)

    A_inf, A_i, k, k_err = fit_absorbance_vs_time_first_order(initial_guess, absorbances, time, 'no')    
    fit_parameters = [A_inf, A_i, k]
    time = np.array(time)

    # Plot
    plt.figure(figsize=(8, 6))

    plt.plot(time, absorbances, 'ko', label='absorbance at {} nm'.format(wavelength))
    plt.plot(time, fitting_function(time, *fit_parameters), color='navy', linestyle='-', label='fitted function')
    plt.xlabel('irradiation time / s')
    plt.ylabel('absorbance')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.legend() 
    plt.show()

def plot_photokinetics_and_fitted_values_first_order(initial_guess, irradiation_times, dataframe, absorbances, wavelength, palette, fontsize_global, fontsize_legend, xmin_1, xmax_1, xmin_2, xmax_2, ymin, ymax):
    
    # Plotting
    plt.figure(figsize=(14, 6))
    plt.rcParams.update({'font.size': fontsize_global})

    # Create a color palette based on the concentrations list
    palette = sns.color_palette(palette, n_colors=len(irradiation_times))

    # Determine the minimum and maximum concentration values
    min_time = min(irradiation_times)
    max_time = max(irradiation_times)

    # First subplot
    plt.subplot(1,2,1)
    for i, column in enumerate(dataframe.columns[1:], start=1):
        x_values = dataframe.iloc[:, 0]  # X-values from the first column
        y_values = dataframe.iloc[:, i]  # Y-values from each subsequent column
        time = irradiation_times[i - 1]  # Get concentration corresponding to the column

        # Scale the concentration value to fit within the range of available colors
        scaled_time = (time - min_time) / (max_time - min_time)
        color_index = int(scaled_time * (len(palette) - 1))  # Scale to fit within palette range
        color = palette[color_index]  # Choose color based on scaled concentration value

        plt.plot(x_values, y_values, label=f"{time} s", color=color)

    plt.xlabel("wavelength / nm")  # Update with your x-axis label
    plt.ylabel("absorbance")  # Update with your y-axis label
    plt.xlim(xmin_1,xmax_1)
    plt.ylim(ymin,ymax)
    plt.legend(fontsize= fontsize_legend)

    # Fitting Data
    time = np.array(irradiation_times)

    # Define fitting function
    def fitting_function(t, A_inf, A_i, k):
        return A_inf +  (A_i - A_inf) * np.exp(-k * t)

    # Perform curve fit
    fit_parameters, covariance = curve_fit(fitting_function, time, absorbances, p0=initial_guess)

    plt.subplot(1,2,2)
    plt.plot(time, absorbances, 'ko', label='absorbance at {} nm'.format(wavelength))
    plt.plot(time, fitting_function(time, *fit_parameters), color='navy', linestyle='-', label='fitted function')
    plt.xlabel('irradiation time / s')
    plt.ylabel('absorbance')
    plt.xlim(xmin_2,xmax_2)
    plt.ylim(ymin,ymax)
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_QY(k_fit, absorbances, c_0, photon_flux, err_photon_flux, err_k, print_option):

    QY = k_fit * c_0 / (photon_flux * (1 - 10**(-absorbances[0])))
    err_QY = np.sqrt( ( c_0 / (photon_flux * (1 - 10**(-absorbances[0]))) * err_k )**2 + ( k_fit * c_0 / (photon_flux**2 * (1 - 10**(-absorbances[0]))) * err_photon_flux )**2 )

    if print_option == 'yes':
        print('quantum yield = {} ± {}'.format(QY, err_QY) )

def calculate_pseudo_total_QY(k_fit, absorbances, thickness_cuvette, photon_flux, err_photon_flux, err_k, print_option):

    photo_kinetic_factor = ( 1 - 10**(-absorbances[0]) ) / absorbances[0]
    Q = k_fit / ( thickness_cuvette * photon_flux * photo_kinetic_factor ) 
    err_Q = np.sqrt( ( 1/( thickness_cuvette * photon_flux * photo_kinetic_factor ) * err_k )**2 + ( - k_fit / ( thickness_cuvette * photon_flux**2 * photo_kinetic_factor ) * err_photon_flux )**2 ) 


    if print_option == 'yes':
        print('pseudo total QY = {:e} ± {:e} L/mol/cm'.format(Q, err_Q))

def get_parameters(testing = False):

    if testing:

        # Get Files
        path = os.getcwd()
        file_UV_Vis = os.path.join(path, r"E:\01_Raw\01_Azobenzene\UV_Vis\251127\251127_AzoAlaNH2_cycling_365_455_645nm.csv")
        file_QYDS = os.path.join(path, r"E:\01_Raw\01_Azobenzene\UV_Vis\251127\251117_C1_365nm.dat")

        # Measurement parameters
        volume = 3 # mL
        conc_initial = 50E-6 # M
        irradiation_wavelength = 365 # nm
        conversion_factor = 0.2315 # mW / V
        voltage_after_ref = 3.29 # V
        kinetics_wavelength = 365 # nm
        kinetics_p0 = [0,0,0]

        # Dataframe parameters
        UV_Vis_last_row = 603
        UV_Vis_first_column = 0
        UV_Vis_last_column = 24

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
            'UV Vis last row' : UV_Vis_last_row,
            'UV Vis first col' : UV_Vis_first_column,
            'UV Vis last col' : UV_Vis_last_column,
        }

def main(testing=False):

    # Get Parameters
    params = get_parameters(testing=testing)
    
    # Create Dataframes
    df_UV_Vis = produce_UV_Vis_dataframe(file=params['UV Vis file'],
                                         last_row_with_data=params['UV Vis last row'],
                                         first_column_with_data=params['UV Vis first col'],
                                         last_column_with_data=params['UV Vis last col'],
                                         print_option=False)
    params['UV Vis df'] = df_UV_Vis
    df_QYDS = producte_QYDS_dataframe(file=params['QYDS file'],
                                      rows_to_delete_indices_list=[],
                                      print_option=False)
    params['QYDS df'] = df_QYDS
    if len(df_QYDS) != len(df_UV_Vis.columns) - 1:
        print('QYDS dataframe and irradiation times have not the same length!')
        return
    
    params = extract_absorption_at_wavelength(params=params)
    
    calculate_photon_flux(params=params)

    params = fit_absorbance_vs_time_first_order(params['Initial guess kinetics'],
                                                params['Kinetics abs'],
                                                df_QYDS.iloc[:,1],
                                                'yes')
    
    plot_photokinetics_and_fitted_values_first_order(initial_guess=params['Initial guess kinetics'], 
                                                irradiation_times= df_QYDS.iloc[:,1],
                                                dataframe= df_UV_Vis,
                                                absorbances= params['Kinetics abs'],
                                                wavelength=365,
                                                palette= 'crest',
                                                fontsize_global= 12,
                                                fontsize_legend= 6,
                                                xmin_1= 250,
                                                xmax_1= 800,
                                                xmin_2= -5,
                                                xmax_2= 610,
                                                ymin= 0,
                                                ymax=0.8
                                                )
    

if __name__ == "__main__":

    main(testing=True)

    # # # Fit dilution series and photokinetics
    # A_inf_par, A_i_par, k_par, err_k_par = fit_absorbance_vs_time_first_order(initial_guess_photokinetics_par, 
    #                                                                           absorbances_QYDS_par, 
    #                                                                           QYDS_irradiation_times_par, 
    #                                                                           'yes')
    # power_density_par = calculate_power_density(voltage_after_reference_cuvette_par, 
    #                                             correction_factor_par, 
    #                                             1, 
    #                                             'yes')
    # photon_flux_par, err_photon_flux_par = calculate_photon_flux(absorbtion_wavelength_par, 
    #                                                              voltage_after_reference_cuvette_par, 
    #                                                              correction_factor_par, 
    #                                                              total_volume, 
    #                                                              0.1, 
    #                                                              0.02,
    #                                                              'yes')
    


    # # Calculate quantum yields
    # calculate_QY(k_par, 
    #              absorbances_QYDS_par, 
    #              initial_concentration_par, 
    #              photon_flux_par, 
    #              err_photon_flux_par, 
    #              err_k_par, 
    #              'yes')
    # calculate_pseudo_total_QY(k_par, 
    #                           absorbances_QYDS_par, 
    #                           thickness_cuvette_par, 
    #                           photon_flux_par, 
    #                           err_photon_flux_par, 
    #                           err_k_par, 
    #                           'yes')