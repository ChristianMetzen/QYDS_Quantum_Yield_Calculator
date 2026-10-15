These scripts calculate photoswitching quantum yields of reversible photoreactions (azobenzenes, DAEs, ...) following Ladányi et al., Photochem. Photobiol. Sci., 2017, 16, 1757.

Thermal relaxation can optionally be included in the QY fit (see step 4).

All scripts read the UV-Vis .csv files both raw and after the csv changer.

## 0. Preparation

"00_UVVis_csv_changer.py": load a UV-Vis csv, tick the spectra you want to remove and save. The metadata block is removed.

"00_get_absorption_coeffs_of_pure_components.py": absorption coefficients of the pure components from two spectra with known composition (e.g. thermal equilibrium and a PSS) and the total concentration. The composition can be determined via NMR or HPLC with DAD. If thermal relaxation is fast, measure the composition and the spectrum at the same time.
  The 1st column of the output is component A, the 2nd column component B. This order defines A and B in the QY script.

## 1. Photon flux

"01a_Ferrioxalate_Actinometry.ipynb"
  Part 1: photon flux from the photodiode (voltage after reference cuvette, wavelength, channel).
  Part 2: required irradiation time for ferrioxalate actinometry.
  Part 3: photon flux from the ferrioxalate measurement (compared with Part 1).
  The photon flux in mol s-1 is the input for the QY script.

## 2. Kinetics dataframe

"02a_kinetics_preparation.py" converts the measured data to the needed structure. UV-Vis files can be used raw or after the csv changer.
  1. Multiple spectra: the first spectrum is taken before irradiation, then one spectrum after every irradiation step.
      Measurement type "Multiple Spectra". Load the UV-Vis file, set the irradiation time of the first spectrum (normally 0) and the step. Times are 0, step, 2*step, ...
  2. Kinetic trace directly in the QYDS under continuous irradiation. An initial spectrum is needed to correct the absorbance offset.
      Measurement type "Kinetics". Load the kinetics file and the initial spectrum, enter the observation wavelength.
  Select the output folder and file name and press "Build & Save Kinetics DataFrame".

## 3./4. Quantum yields

"03_QY_calculation.py"
  Load the kinetics dataframe, the absorption coefficients and select a results folder.
  Enter the LED (or central wavelength and FWHM), the photon flux in mol s-1 and the irradiated volume.
  Concentrations: for spectra series they are obtained by a two-component least-squares fit in the "Concentration WL" range, so no total concentration is needed. The entered concentration is only used for single-wavelength kinetics.
  Photoreaction: "forward + backward" fits both QYs, "forward only" fixes the QY of the back reaction to 0 (e.g. irradiation where the product does not absorb).
  Thermal relaxation: choose the direction ("2nd -> 1st column" is cis -> trans for a trans/cis file) and enter the rate constant k = ln2 / t1/2 in s-1. k is fixed, not fitted; determine it separately in the dark at the same temperature.
  Note: the time axis is the irradiation time. Thermal relaxation during dark periods (e.g. while a spectrum is recorded with the LED off) is not included.
  Fitted parameters: both QYs and the initial composition. Uncertainties are 1 sigma from the fit and do not include the photon flux error.
