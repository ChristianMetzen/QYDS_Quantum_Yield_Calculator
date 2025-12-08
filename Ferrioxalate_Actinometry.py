import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from scipy import constants

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

def extract_absorption_at_wavelength(dataframe,
                                     wavelength=510):

    # Find the row index closest to the specified wavelength
    closest_row_index = (dataframe.iloc[:, 0] - wavelength).abs().idxmin()
    
    # Extract absorption values at the closest row index
    absorption_values = dataframe.iloc[closest_row_index, 1:].tolist()

    return absorption_values

def gui():

    def calculate_photon_flux():
        try:
            wavelength = float(wavelength_var.get())
            time = float(time_var.get())
        except ValueError:
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, "Error: Please enter valid numeric values.")
            return  # stop here

        # Lookup table for fixed wavelengths
        wavelength_settings = {
            405: (6, 1.14),
            436: (6, 1.11),
            458: (10, 1.12),
            488: (80, 1.08),
            514: (200, 0.93),
        }

        # Determine conc and QY
        if 254 <= wavelength <= 366:
            conc, QY = 6, 1.23
        elif wavelength in wavelength_settings:
            conc, QY = wavelength_settings[wavelength]
        else:
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, "Error: No settings found for this wavelength.")
            return  # stop so no further code runs
        
        V1, V2, V3, abs_coeff, d = 3, 1, 10, 11100, 1

        try:
            df = produce_UV_Vis_dataframe(file_UV_Vis_var.get())
            absorbances = extract_absorption_at_wavelength(df)
            dA = absorbances[1] - absorbances[0]
        except:
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, "Error: Please select a valid file.")

        try:
            photon_flux = dA * V1 * V3 / ( QY * abs_coeff * V2 * d * time ) * 1E-3
        except:
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, "Error in photon flux calculation.")

        # Output results
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Input Parameters:\n")
        output_box.insert(tk.END, f"  Irradiation wavelength: {wavelength} nm\n")
        output_box.insert(tk.END, f"  Irradiation time: {time} s\n")
        output_box.insert(tk.END, "Used Parameters:\n")
        output_box.insert(tk.END, f"  Concentration: {conc} mM\n")
        output_box.insert(tk.END, f"  Quantum Yield: {QY}\n")
        output_box.insert(tk.END, f"  Irradiated volume: {V1} mL\n")
        output_box.insert(tk.END, f"  Transferred volume: {V2} mL\n")
        output_box.insert(tk.END, f"  Total analytic volume: {V3} mL\n")
        output_box.insert(tk.END, f"  Absorption coefficient at 510 nm: {abs_coeff} L mol⁻¹ cm⁻¹\n")
        output_box.insert(tk.END, f"  Optical pathlength: {d} cm\n")
        output_box.insert(tk.END, "Output:\n")
        output_box.insert(tk.END, f"  Photon flux: {photon_flux:.2e} mol s⁻¹")

    

    root = tk.Tk()
    root.title("Ferrioxalate Actinometry Evaluation")

    def choose_file(var):
        file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        var.set(file)

    file_UV_Vis_var = tk.StringVar()
    ttk.Label(root, text="UV Vis file:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    ttk.Button(root, text="Browse", command=lambda: choose_file(file_UV_Vis_var)).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

    # --- Wavelength ---
    ttk.Label(root, text="Irradiation wavelength / nm:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    wavelength_var = tk.StringVar(value='365')
    ttk.Entry(root, textvariable=wavelength_var).grid(row=1, column=1, padx=10, pady=10)

    # --- Voltage ---
    ttk.Label(root, text="Irradiation time / s:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    time_var = tk.StringVar(value='400')
    ttk.Entry(root, textvariable=time_var).grid(row=2, column=1, padx=10, pady=10)

    # --- Calculate Button ---
    ttk.Button(root, text="Calculate Photon Flux", command=calculate_photon_flux).grid(row=4, column=0, columnspan=2, pady=15)

    # --- Output Box ---
    output_box = tk.Text(root, width=80, height=20)
    output_box.grid(row=6, column=0, columnspan=3, padx=10, pady=20)

    root.mainloop()

if __name__ == "__main__":
    gui()