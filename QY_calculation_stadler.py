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

def produce_UV_Vis_dataframe(file, last_row_with_data, first_column_with_data, last_column_with_data, print_option):

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

    if print_option == 'yes':
        print(df)

    return df

def producte_QYDS_dataframe(file, rows_to_delete_indices_list, print_option):
    data = {'Runtime': [], 'Illum Time': [], 'Cell Value': []}

    with open(file, 'r') as file:
        lines = file.readlines()

        # Extract data from the specified column    
        for line in lines:
            if not line.startswith("#"): 
                line_data = line.split("\t")
                if len(line_data) >= 7:
                    try:
                        data['Runtime'].append(float(line_data[3].replace(",", ".")))
                        data['Illum Time'].append(float(line_data[4].replace(",", ".")))
                        data['Cell Value'].append(float(line_data[6].replace(",", ".")))
                    except ValueError:
                        print("Error converting to float:", line_data)

    df = pd.DataFrame(data)
    df = df.drop(rows_to_delete_indices_list)
    df = df.astype(float)

    if print_option == 'yes':
        print(df)
    
    return df

def check_if_QYDS_file_and_csv_file_have_same_length(QYDS_dataframe, UV_Vis_dataframe):

    if len(QYDS_dataframe) != len(UV_Vis_dataframe.columns) - 1:
        print('QYDS dataframe and irradiation times have not the same length!')

def calculate_power_density(voltage_after_reference_cuvette, correction_factor, area_in_square_cm, print_option):
    
    power_density = voltage_after_reference_cuvette * correction_factor / area_in_square_cm * 10 # W m-2

    if print_option == 'yes':
        print('Power density = {:.5f} W m-2'.format(power_density))

    return power_density

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

def extract_absorption_at_wavelength(dataframe, wavelength):
    # Find the row index closest to the specified wavelength
    closest_row_index = (dataframe.iloc[:, 0] - wavelength).abs().idxmin()
    
    # Extract absorption values at the closest row index
    absorption_values = dataframe.iloc[closest_row_index, 1:].tolist()


    return absorption_values

def fit_absorbance_vs_time_first_order(initial_guess, absorbances, time, print_option):

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

def calculate_photon_flux(wavelength, voltage_after_reference, correction_factor, volume, voltage_error, length_error, print_option):

    photon_flux = wavelength * voltage_after_reference * correction_factor / (constants.Planck * constants.speed_of_light * constants.Avogadro * volume) * 1E-12
    power_error = voltage_error * correction_factor * 1E-3
    volume_error = 0.1 * 0.1 * length_error
    err_photon_flux = np.sqrt( ( wavelength*1E-9 / ( constants.Planck*constants.speed_of_light*constants.Avogadro*volume ) * power_error )**2 + ( - wavelength*1E-9 * voltage_after_reference*correction_factor*1E-3 / ( constants.Planck*constants.speed_of_light*constants.Avogadro*volume**2 ) * volume_error )**2 )

    if print_option == 'yes':
        print('Photon flux = {} ± {} mol / s / L'.format(photon_flux, err_photon_flux))

    return photon_flux, err_photon_flux

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

if __name__ == "__main__":

    # Parameters
    absorbtion_wavelength_par = 365 # nm
    correction_factor_par = 0.2465
    voltage_after_reference_cuvette_par = 8.01
    thickness_cuvette_par = 1 # cm
    initial_concentration_par = 285.7143E-6 # mol L-1
    initial_guess_photokinetics_par = [1, 1, 1]
    total_volume = 0.003 # L

    # Get files
    path = os.getcwd()
    file_UV_Vis_irradiation = os.path.join(path, r'2_files\251008.csv')
    file_QYDS = os.path.join(path, r'2_files\251007.dat')

    # Create dataframes
    df_UV_Vis_irradtiation = produce_UV_Vis_dataframe(file_UV_Vis_irradiation, 153, 0, 28, 'no')
    df_QYDS = producte_QYDS_dataframe(file_QYDS, [], 'no')
    check_if_QYDS_file_and_csv_file_have_same_length(df_QYDS, df_UV_Vis_irradtiation)
    QYDS_irradiation_times_par = df_QYDS['Illum Time'].tolist()

    # # Extract absorption
    absorbances_QYDS_par = extract_absorption_at_wavelength(df_UV_Vis_irradtiation, absorbtion_wavelength_par)

    # # Fit dilution series and photokinetics
    A_inf_par, A_i_par, k_par, err_k_par = fit_absorbance_vs_time_first_order(initial_guess_photokinetics_par, 
                                                                              absorbances_QYDS_par, 
                                                                              QYDS_irradiation_times_par, 
                                                                              'yes')
    power_density_par = calculate_power_density(voltage_after_reference_cuvette_par, 
                                                correction_factor_par, 
                                                1, 
                                                'no')
    photon_flux_par, err_photon_flux_par = calculate_photon_flux(absorbtion_wavelength_par, 
                                                                 voltage_after_reference_cuvette_par, 
                                                                 correction_factor_par, 
                                                                 total_volume, 
                                                                 0.1, 
                                                                 0.02,
                                                                 'yes')

    # Calculate quantum yields
    calculate_QY(k_par, 
                 absorbances_QYDS_par, 
                 initial_concentration_par, 
                 photon_flux_par, 
                 err_photon_flux_par, 
                 err_k_par, 
                 'yes')
    # calculate_pseudo_total_QY(k_par, 
    #                           absorbances_QYDS_par, 
    #                           thickness_cuvette_par, 
    #                           photon_flux_par, 
    #                           err_photon_flux_par, 
    #                           err_k_par, 
    #                           'yes')