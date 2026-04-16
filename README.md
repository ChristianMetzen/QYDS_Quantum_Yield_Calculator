These scripts allow the calculation of photoswitching quantum yields of reversible photoreaction, like azobenzenes or DAEs. 

In the current state, the thermal relaxation is not considered, which is valid for azobenzenes and DAEs.

If you want to measure the photon flux via actiometry with the "00_Ferrioxalate_Actinometry.ipynb" notebook, but the calculated photon fluxes from the solar cell were in good agreement with the measured ones, so this is not necessary.

The scripts use the .dat file from the QYDS and the .csv from the UV Vis as they are. Previous modifications might lead to errors. 

For modification of the UV Vis file, use the "00_UVVis_csv_changer.py"
  Here, you can load a csv and select a first and last column that should be in the final file. Currently, removal of singular columns is not possible with this script. It also removes the metadata block in the csv. 
  Currently, this script runs on the old read-in function for csv files, so bigger measurements could take some time to load

You can plot you inital spectra using the "00_plot_initial_spectra.ipynb" notebook, and also select a wavelength at which the kinetcs are plotted. 

For calculation of the quantum yield, the absorption coefficients of the pure components must be determined. This can be done with the "01_get_absorption_coeffs_of_pure_components.py" script.
  To use this, you need two spectra with known composition and the total concentration. For this, the spectrum of the thermal equilibrium and some PSS is recommended. 
  The composition can be determined via NMR or HPLC with a DAD. If the thermal relaxation is negligible, no corrections are needed here.

Before the calculation of the qunatum yield, the measured data must be converted to the needed structure using the "02_kinetics_preparation.py" script.
  This works for the following 3 types of measurements:
    1. Multiple individual spectra are recorded.
      1a. The cuvette is transferred from the QYDS to the UV Vis at certain irradiation times. In this case the QYDS .dat file must be used to get the irradiation times. Here it is important the the entries in the .dat file match the number of spectra.
        Set the Measurement type to "Multiple spectra" and the Manual / Auto parameter to "manual". Load the UV-Vis file and the QYDS file. For this, the irradiation wavelength is not important.
      1b. The spectra were measured directly in the QYDS under continuous irradiation. (Note, that the time point of the whole spectra will be set to the start of the measurement. A delay between the wavelengths is not considered here). 
        Set the Measurement type to "Multiple spectra" and the Manual / Auto parameter to "auto". Load the UV-Vis file and set the irradiation time at the start and the time-step between the measurements. For this, the irradiation wavelength is not important.
    2. A kinetic measurement directly in the QYDS under continuous irradiation. Note, that the observation wavelength must be the irradiation wavelength, for the qunatum yield calculation to work. Also an initial spectrum must be measured to correct the absorbance.
      Set the Measurement type to "Kinetics". Load the UV-Vis files (Kinetics file and Initial UV-Vis spectrum). Enter the irradiation/observation wavelength. 
  Select the output folder and the file name and press "Build & Save Kinetics DataFrame".

  To calculate the quantum yields, use the "03_QY_calculation.py" script.
    Load the Kinetics dataframe form the step before, the absorption coefficient and select a results folder.
    Enter the irradiation wavelength, the voltage after reference cuvette, the volume, the concentration and the channel. From this, the photon flux is automatically calculated.
    Press "Run Analysis" and adjusted the plotting parameters as needed.
    
  
