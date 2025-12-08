import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from scipy import constants

def get_correction_factor(wavelength=365,
                          channel='low'):
    
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

    df = pd.DataFrame(data)

    correction_factor = np.interp(wavelength, data['wavelength'], data[channel])
    
    return correction_factor


def gui():

    def calculate_time():
        try:
            wavelength = float(wavelength_var.get())
            channel = channel_var.get()
            voltage = float(voltage_var.get())
            absorbance = float(absorbance_var.get())
            correction_factor = get_correction_factor(wavelength=wavelength,
                                                      channel=channel)
            photon_flux = wavelength * voltage * correction_factor / (constants.Planck * constants.speed_of_light * constants.Avogadro) * 1E-12


            result_time = (absorbance*3*10)/(1.20*11100*1*1*photon_flux)*1E-3

            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, f"Input Parameters:\n")
            output_box.insert(tk.END, f"  Wavelength: {wavelength} nm\n")
            output_box.insert(tk.END, f"  Channel: {channel}\n")
            output_box.insert(tk.END, f"  Voltage after reference: {voltage} V\n")
            output_box.insert(tk.END, f"  Absorbance: {absorbance}\n")
            output_box.insert(tk.END, f"Output:\n")
            output_box.insert(tk.END, f"  Correction factor: {correction_factor} mW / V\n")
            output_box.insert(tk.END, f"  Assumed Photon flux: {photon_flux:.2e} mol / s\n")
            output_box.insert(tk.END, f"  Calculated Time: {result_time:.0f} s\n")

        except ValueError:
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, "Error: Please enter valid numeric values.")

    root = tk.Tk()
    root.title("Irradiation Time Calculator")

    # --- Wavelength ---
    ttk.Label(root, text="Wavelength / nm:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    wavelength_var = tk.StringVar(value='365')
    ttk.Entry(root, textvariable=wavelength_var).grid(row=0, column=1, padx=10, pady=10)

    # --- Channel ---
    ttk.Label(root, text="Channel:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    channel_var = tk.StringVar()
    ttk.Combobox(
        root,
        textvariable=channel_var,
        values=["high", "low"],
        state="readonly"
    ).grid(row=1, column=1, padx=10, pady=10)
    channel_var.set("low")

    # --- Voltage ---
    ttk.Label(root, text="Voltage after reference / V:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    voltage_var = tk.StringVar(value='7.58')
    ttk.Entry(root, textvariable=voltage_var).grid(row=2, column=1, padx=10, pady=10)

    # --- Absorbance ---
    ttk.Label(root, text="Desired absorbance:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
    absorbance_var = tk.StringVar(value='0.5')
    ttk.Entry(root, textvariable=absorbance_var).grid(row=3, column=1, padx=10, pady=10)

    # --- Calculate Button ---
    ttk.Button(root, text="Calculate Irradiation Time", command=calculate_time).grid(row=4, column=0, columnspan=2, pady=15)

    # --- Output Box ---
    output_box = tk.Text(root, width=80, height=20)
    output_box.grid(row=6, column=0, columnspan=3, padx=10, pady=20)

    root.mainloop()

if __name__ == "__main__":
    gui()