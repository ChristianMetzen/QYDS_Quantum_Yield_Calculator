import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

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
        else:
            df = df.iloc[1:]

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
    A_PSS = dataframe_PSS.iloc[:,1].to_numpy()

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

        ax.set_xlabel("wavelength / nm")
        ax.set_ylabel("absorption coefficient / L mol⁻¹ cm⁻¹")
        ax.legend()

        canvas.draw()

    # =========================
    # Layout
    # =========================
    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(side="left", fill="y")

    plot_frame = ttk.Frame(root, padding=10)
    plot_frame.pack(side="right", fill="both", expand=True)

    # ---- File ----
    ttk.Label(control_frame, text="UV-Vis spectrum thermal equilibrium").grid(row=0, column=0, sticky="w")
    ttk.Entry(control_frame, textvariable=file_teq_var, width=40).grid(row=1, column=0, pady=2)
    ttk.Button(control_frame, text="Select UV-Vis spectrum thermal equilibrium", command=lambda: choose_file(file_teq_var)).grid(row=1, column=1, padx=5)

    # ---- File ----
    ttk.Label(control_frame, text="UV-Vis file PSS").grid(row=2, column=0, sticky="w")
    ttk.Entry(control_frame, textvariable=file_PSS_var, width=40).grid(row=3, column=0, pady=2)
    ttk.Button(control_frame, text="Select UV-Vis file PSS", command=lambda: choose_file(file_PSS_var)).grid(row=3, column=1, padx=5)

    # ---- Total concentration ----
    ttk.Label(control_frame, text="Total concentration c(tot) / uM").grid(row=4, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=ctot_var, width=15).grid(row=5, column=0, sticky="w")

    # ---- x(teq) ----
    ttk.Label(control_frame, text="Percentage Component B in thermal equilibrium").grid(row=6, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=x_teq_var, width=15).grid(row=7, column=0, sticky="w")

    # ---- x(PSS) ----
    ttk.Label(control_frame, text="Percentage Component B in PSS").grid(row=8, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=x_PSS_var, width=15).grid(row=9, column=0, sticky="w")

    # ---- Component names ----
    ttk.Label(control_frame, text="Component A name").grid(row=10, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=name1_var, width=20).grid(row=11, column=0)

    ttk.Label(control_frame, text="Component B name").grid(row=12, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(control_frame, textvariable=name2_var, width=20).grid(row=13, column=0)

    # ---- Buttons ----
    ttk.Button(control_frame, text="Run Analysis", command=run_analysis)\
        .grid(row=14, column=0, pady=(20, 5), sticky="ew")

    ttk.Button(control_frame, text="Save Results", command=save_results)\
        .grid(row=15, column=0, sticky="ew")

    # =========================
    # Plot canvas
    # =========================
    fig = plt.Figure(figsize=(16, 12), dpi=100)  
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":

    run_gui()