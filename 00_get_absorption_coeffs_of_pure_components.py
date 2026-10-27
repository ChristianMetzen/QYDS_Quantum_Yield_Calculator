import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import csv

def read_in_UV_Vis_csv(file):
    """Read a Cary UV-Vis csv, raw or cleaned by 00_UVVis_csv_changer.

    Layout: row 0 = spectrum names (every 2nd column), row 1 = 'Wavelength (nm),Abs',
    then data as x/y column pairs, optionally followed by an empty line and the
    instrument metadata. Works for spectra and for kinetics files (x = time).
    Returns a DataFrame: column 0 = x of the first spectrum, then one column per spectrum.
    """
    with open(file, 'rb') as f:
        text = f.read().decode('utf-8', errors='replace')
    text = text.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')

    def is_num(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    header, data = [], []
    for row in csv.reader(text.split('\n')):
        if not any(c.strip() for c in row):     # empty or ',,,' line
            if data:
                break                            # metadata block starts
            continue
        if row and is_num(row[0].strip()):
            data.append(row)
        elif data:
            break
        else:
            header.append(row)
    if not data:
        raise ValueError(f"No numeric data found in {os.path.basename(file)}")

    n_cols = max(len(r) for r in data)
    arr = np.full((len(data), n_cols), np.nan)
    for i, r in enumerate(data):
        for j, c in enumerate(r):
            if c.strip() and is_num(c.strip()):
                arr[i, j] = float(c)

    y_idx = [j for j in range(1, n_cols, 2) if not np.all(np.isnan(arr[:, j]))]
    names = []
    for j in y_idx:
        name = header[0][j - 1].strip() if header and len(header[0]) > j - 1 else ''
        names.append(name or f'spectrum {len(names) + 1}')
    names = [f'{n} ({i})' if names.count(n) > 1 else n for i, n in enumerate(names)]

    df = pd.DataFrame(arr[:, y_idx], columns=names)
    df.insert(0, 'x', arr[:, 0])
    return df.dropna(subset=['x']).reset_index(drop=True)


def calculate_absorptin_coefficients(dataframe_teq = None,
                                     dataframe_PSS = None,
                                     ratio_teq = 0.1,
                                     ratio_PSS = 0.9,
                                     names_list = ['1', '2'],
                                     c_tot = 100):
    
    c_tot = c_tot*1E-6
    x_teq = ratio_teq
    x_PSS = ratio_PSS
    d = 1
    A_teq = dataframe_teq.iloc[:,1].to_numpy()
    # PSS spectrum on the wavelength grid of the teq spectrum
    pss   = dataframe_PSS.sort_values(dataframe_PSS.columns[0])
    A_PSS = np.interp(dataframe_teq.iloc[:,0].to_numpy(),
                      pss.iloc[:,0].to_numpy(), pss.iloc[:,1].to_numpy())

    # Normalize absorbances

    y_teq = A_teq / (c_tot*d)
    y_PSS = A_PSS / (c_tot*d)

    # Solve for pure spectra
    den = x_PSS - x_teq

    e_A = (x_PSS * y_teq - x_teq * y_PSS) / den
    e_B = ((1 - x_teq) * y_PSS - (1 - x_PSS) * y_teq) / den

    df_abs = pd.DataFrame({
        'wavelength / nm' : dataframe_teq.iloc[:,0],
        names_list[0] : e_A,
        names_list[1] : e_B
    })


    return df_abs


def run_gui():

    root = tk.Tk()
    root.title("UV-Vis PSS Analysis")
    root.geometry("1280x720")

    # =========================
    # Variables
    # =========================
    file_teq_var = tk.StringVar(value=r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\example_files\Multiple_spectra_manual\Spectrum_thermal_equilibrium.csv")
    file_PSS_var = tk.StringVar(value=r"C:\Users\metzen\sciebo\Promotion\10_Scripts\00_Github\QYDS_Quantum_Yield_Calculator\example_files\Multiple_spectra_manual\Spectrum_PSS.csv")
    name1_var = tk.StringVar(value="trans")
    name2_var = tk.StringVar(value="cis")
    ctot_var = tk.StringVar(value="40") 
    x_teq_var = tk.StringVar(value='5.6')
    x_PSS_var = tk.StringVar(value='91')

    df_abs_result = {"df": None}

    # =========================
    # File selection
    # =========================
    def choose_file(var):
        file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file:
            var.set(file)

    # =========================
    # Analysis
    # =========================
    def run_analysis():
        try:
            x_teq = float(x_teq_var.get())/100
            x_PSS = float(x_PSS_var.get())/100
            c_tot = float(ctot_var.get())

            df_teq = read_in_UV_Vis_csv(file=file_teq_var.get())
            df_PSS = read_in_UV_Vis_csv(file=file_PSS_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid ratios or c_tot.")
            return

        df_abs = calculate_absorptin_coefficients(
            dataframe_teq=df_teq,
            dataframe_PSS=df_PSS,
            ratio_teq=x_teq,
            ratio_PSS=x_PSS,
            names_list=[name1_var.get(), name2_var.get()],
            c_tot=c_tot 
        )

        df_abs_result["df"] = df_abs
        plot_absorption(df_abs)

    # =========================
    # Save results
    # =========================
    def save_results():
        if df_abs_result["df"] is None:
            messagebox.showerror("Error", "No results to save.")
            return

        input_path = file_teq_var.get()
        folder = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]

        save_path = os.path.join(folder, f"{base}_absorption_coefficients.csv")
        df_abs_result["df"].to_csv(save_path, index=False)

        print(f"Saved absorption coefficients to:\n{save_path}")

    # =========================
    # Plotting
    # =========================
    def plot_absorption(df_abs):
        fig.clear()
        ax = fig.add_subplot(111)

        ax.plot(df_abs.iloc[:, 0], df_abs.iloc[:, 1], label=df_abs.columns[1])
        ax.plot(df_abs.iloc[:, 0], df_abs.iloc[:, 2], label=df_abs.columns[2])
        ax.axhline(0, color='gray', ls='--')

        ax.set_xlabel("wavelength / nm")
        ax.set_ylabel("absorption coefficient / L mol⁻¹ cm⁻¹")
        ax.legend()

        canvas.draw()

    # =========================
    # Layout
    # =========================
    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(side="left", fill="y")
    control_frame.grid_columnconfigure(0, weight=1)

    plot_frame = ttk.Frame(root, padding=10)
    plot_frame.pack(side="right", fill="both", expand=True)

    # ---- File A ----
    ttk.Label(control_frame, text="UV-Vis spectrum A (time point A)").grid(row=0, column=0, sticky="w")
    ttk.Entry(control_frame, textvariable=file_teq_var, width=60).grid(row=1, column=0, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Browse", command=lambda: choose_file(file_teq_var), width=10).grid(row=1, column=1, padx=5)

    # ---- File B ----
    ttk.Label(control_frame, text="UV-Vis spectrum B (time point B)").grid(row=2, column=0, sticky="w")
    ttk.Entry(control_frame, textvariable=file_PSS_var, width=60).grid(row=3, column=0, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Browse", command=lambda: choose_file(file_PSS_var), width=10).grid(row=3, column=1, padx=5)

    # ---- Total concentration ----
    ttk.Label(control_frame, text="Total concentration c(tot) / µM").grid(row=4, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=ctot_var, width=15).grid(row=5, column=0, sticky="w")

    # ---- Note on Component B ----
    ttk.Label(control_frame,
              text="Note: Component B is defined as the species predominant at time point B.",
              foreground="gray").grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 2))

    # ---- x(teq) ----
    ttk.Label(control_frame, text="Percentage of Component B at time point A / %").grid(row=7, column=0, sticky="w", pady=(4, 0))
    ttk.Entry(control_frame, textvariable=x_teq_var, width=15).grid(row=8, column=0, sticky="w")

    # ---- x(PSS) ----
    ttk.Label(control_frame, text="Percentage of Component B at time point B / %").grid(row=9, column=0, sticky="w", pady=(4, 0))
    ttk.Entry(control_frame, textvariable=x_PSS_var, width=15).grid(row=10, column=0, sticky="w")

    # ---- Component names ----
    ttk.Label(control_frame, text="Component A name").grid(row=11, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=name1_var, width=20).grid(row=12, column=0)

    ttk.Label(control_frame, text="Component B name").grid(row=13, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=name2_var, width=20).grid(row=14, column=0)

    # ---- Buttons ----
    ttk.Button(control_frame, text="Run Analysis", command=run_analysis)\
        .grid(row=15, column=0, pady=(20, 5), sticky="ew")

    ttk.Button(control_frame, text="Save Results", command=save_results)\
        .grid(row=16, column=0, sticky="ew")

    # =========================
    # Plot canvas
    # =========================
    fig = plt.Figure(figsize=(16, 12), dpi=100)  
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":

    run_gui()