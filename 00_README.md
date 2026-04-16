This script allows you to calculate the photoswitching Quantum Yields. The following cases have been tested:
a) Azobenzene: A reversible photoreaction, where during irradiation three processes happen:
  1. trans -> cis isomerization
  2. cis -> trans backisomerization
  3. thermal cis -> trans isomerization

Currently, the use of the Ladanyi method is recommended. In principle, the Stadler method can be used without the knowledge of the absorption coefficient of the pure compound. However, it is limited to an absorbance below 0.1. Also, for azobenzene, only a pseudo total 
quantum yield can be determined and the individual quantum yield remain unknown.
For the Ladanyi method, the absorption coefficients of the pure spectra need to be known. This can be done with two methods.
a) Determining the ratio of A/B in the thermal equilibrium and in the PSS. Then the pure spectra can be recovered from the UV Vis spectra in the thermal equilibrium and the PSS.
b) When one species predominates in the thermal equilibrium (~99% for azobenzene), the spectrum in the thermal equilibrium can be used as the pure spectrum of the predominent species. The pure spectrum of the other species can be calculated when a wavelength exists, where
the absorption coefficent of the predominant species is significantly higher, then that of the other species. For this, the script calculate_absorption_coefficients_of_pure_components.py can be used. The input needs to be a csv file in the format from the Agilent Cary 60,
where the first spectrum is the spectrum in the thermal equlibrium and the last spectrum is the PSS. Select this file and enter the wavelength, at which the predominant species has a significant higher absorption coefficient. Also the total concentration is needed, 
to calculate the absorption coefficients. Enter the the names of the components and press "Run Analyis". The names that you give here, will also be used in the QY script. When the results are satisfactory you can save the results. 

The programm Facpack can be used to evaluate the right wavelength and the resulting spectra. For this, the spectral time series (in a csv in the format of the Agilent Cary 60), needs to be converted in a .mat file. For this, the generate_mat_file.py script can be used. 
Select the csv file, and enter a minimum and/or maximum wavelength if desired. Select input data (D, x, t) and press "Start Conversion".

In Facpack version 1.30 alpha, select under "Area of Feasible Solutions" the "Polygon Inflation (2 & 3 components)". Press "Load" and select the .mat file. On the right, select two components and press "initial NMF". If needed, select choose 3 singular vecors 
(singl vcts:). Then, on the bottom left, press "Compute AFS". Now, on the bottom right, you can press "AFS 1" to see which spectra are feasible for your component 1 and press "AFS 2" to see which spectra are feasible for component 2. For photoreactions, where both 
components have similar and/or overlapping spectra (as for azobenzene) you can expect a wide range of feasible solutions. However, this might help to determine the right wavelength for the calculation of the pure spectra.

Now, you can calculate the photoswitching quantum yield this the QY_calculation.py script. On the top you can switch between the two methods. On the top, there are specific parameters for every method and below that are shared parameters.
The shared (general) parameters consists of the file selection. 
1.If you measured a specturm for every time step, seleect the csv file and set the "Measurement:" parameter to "Multiple spectra". Also, the QYDS .dat file is needed. Make sure, that for every entry in the
QYDS file, there is one spectrum in the csv file. Remove, if needed, rows from the QYDS file or spectra from the csv. For the latter, the script UVVis_csv_changer.py can be used. Select "Choose CSV file". Then you can the the columns of the csv file and the respective
index on the top. Enter the index of the first column and of the last column and press "Apply Filter". Then save the file with "Save CSV".
2. If you measured the kinetics directly, select the Kinetics file and also select a csv file that contains the initial spectrum. Then set the "Measurement:" parameter to kinetics. If you want to use the Ladanyi method, the kinetic wavelength, needs to match the
irradiation wavelength. 
3. If you measured multiple spectra, but measured them automatically wihtin the QYDS while irradiating select "auto" for the "Manual/Auto" parameter and enter the start time, the stop time and the step-time. This option is currently not tested.
4. The other Global parameters are the irradiation wavelength, the channel (high or low) in which was QYDS was used, the voltage after reference cuvette, the volume and the initial/total concentration.
5. On the right you have the plotting parameters. If set to "0", the plots will scale the axes itself.
6. For the Stadler method, you can either calculate the QY or the pseudo total quantum yield (QY only for irreversible photoreactions). Enter the wavelength at which you want to fit the absorbance vs time, or the wavelength that was used if you measured kinetics directly. 
Also you might change the inital guess of the fit.
7. For the Ladanyi method, you need to select the file with the absorption coefficients of the pure spectra. Also here, you can change the initial guess for the fit. 

If the settings are complete, press the "Start Stadler Analysis" or the "Start Ladanyi Analysis". You will get the results in the output box and see some plots below.
For Stadler you see the time series (if multiple spectra are recoreded) and the fit of the absorbance vs time,
For Ladanyi you see the time series (if multiple spectra are recoreded), the concentration profiles and the fitted concentration vs time. Here, always the species with decreasing concentration is fitted.
