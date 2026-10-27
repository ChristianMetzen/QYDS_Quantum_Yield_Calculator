import csv
import os

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ─────────────────────────────────────────────────────────────────────────────
# Data processing functions
# ─────────────────────────────────────────────────────────────────────────────

def read_UV_Vis_csv(file):
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


def build_kinetics_df_multiple_spectra(uv_vis_file, start=0.0, step=30.0):
    """Spectra recorded at start, start+step, start+2*step, ... s of irradiation."""
    df = read_UV_Vis_csv(uv_vis_file)
    n_spectra = df.shape[1] - 1
    times = start + step * np.arange(n_spectra)
    df.columns = ['wavelength / nm'] + [f'{t:g}' for t in times]
    return df


def build_kinetics_df_kinetics(kinetics_file, uv_vis_start_file, irradiation_wavelength):
    """Single-row kinetics DataFrame from a kinetics trace + initial spectrum.

    The trace is shifted so that its first point matches the initial spectrum at
    the observation wavelength.
    """
    df_kin     = read_UV_Vis_csv(kinetics_file)
    times      = df_kin.iloc[:, 0].to_numpy(dtype=float)
    absorbance = df_kin.iloc[:, 1].to_numpy(dtype=float)

    df_init = read_UV_Vis_csv(uv_vis_start_file)
    wl_idx  = (df_init.iloc[:, 0] - irradiation_wavelength).abs().to_numpy().argmin()
    A_ini   = df_init.iloc[wl_idx, 1]
    absorbance = absorbance + (A_ini - absorbance[0])

    df_kinetics = pd.DataFrame(data=[absorbance], columns=[f'{t:g}' for t in times])
    df_kinetics.insert(0, 'wavelength / nm', irradiation_wavelength)
    return df_kinetics


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():

    root = tk.Tk()
    root.title("Kinetics DataFrame Builder")
    root.resizable(False, False)

    pad = {'padx': 8, 'pady': 4}

    def choose_file(var):
        path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if path:
            var.set(path)

    def choose_folder(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _float(s):
        return float(str(s).replace(',', '.'))

    # ── Measurement type ──────────────────────────────────────────────────────
    type_frame = ttk.LabelFrame(root, text="Measurement Type", padding=8)
    type_frame.grid(row=0, column=0, sticky="ew", **pad)
    type_frame.grid_columnconfigure(1, weight=1)

    measurement_var = tk.StringVar(value='Multiple Spectra')
    ttk.Label(type_frame, text="Measurement type:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        type_frame, textvariable=measurement_var,
        values=['Multiple Spectra', 'Kinetics'],
        state='readonly', width=22
    ).grid(row=0, column=1, sticky="ew", padx=5)

    # ── File selection ────────────────────────────────────────────────────────
    file_frame = ttk.LabelFrame(root, text="File Selection", padding=8)
    file_frame.grid(row=1, column=0, sticky="ew", **pad)
    file_frame.grid_columnconfigure(0, weight=1)

    def file_row(parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row*2, column=0, sticky="w", pady=(6,0))
        f = ttk.Frame(parent)
        f.grid(row=row*2+1, column=0, sticky="ew", pady=2)
        f.grid_columnconfigure(0, weight=1)
        ttk.Entry(f, textvariable=var).grid(row=0, column=0, sticky="ew", padx=(0,5))
        ttk.Button(f, text="Browse", command=lambda: choose_file(var)).grid(row=0, column=1)

    var_uv_mult  = tk.StringVar()
    var_kinetics = tk.StringVar()
    var_uv_start = tk.StringVar()

    file_row(file_frame, "UV-Vis file (Multiple Spectra)",      var_uv_mult,  0)
    file_row(file_frame, "Kinetics file (Kinetics mode)",       var_kinetics, 1)
    file_row(file_frame, "Initial UV-Vis file (Kinetics mode)", var_uv_start, 2)

    # ── Parameters ───────────────────────────────────────────────────────────
    param_frame = ttk.LabelFrame(root, text="Parameters", padding=8)
    param_frame.grid(row=2, column=0, sticky="ew", **pad)
    param_frame.grid_columnconfigure(1, weight=1)

    var_wl    = tk.StringVar(value='365')
    var_start = tk.StringVar(value='0')
    var_step  = tk.StringVar(value='30')

    rows = [
        ("Observation wavelength / nm (Kinetics):",            var_wl),
        ("Irradiation time of 1st spectrum / s (Multiple):",   var_start),
        ("Irradiation step between spectra / s (Multiple):",   var_step),
    ]
    for i, (lbl, var) in enumerate(rows):
        ttk.Label(param_frame, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=var, width=12).grid(row=i, column=1, sticky="w", padx=5)

    # ── Output ───────────────────────────────────────────────────────────────
    out_frame = ttk.LabelFrame(root, text="Output", padding=8)
    out_frame.grid(row=3, column=0, sticky="ew", **pad)
    out_frame.grid_columnconfigure(0, weight=1)

    var_out_folder = tk.StringVar()
    var_out_name   = tk.StringVar(value='kinetics_df.csv')

    ttk.Label(out_frame, text="Save folder:").grid(row=0, column=0, sticky="w")
    f_out = ttk.Frame(out_frame)
    f_out.grid(row=1, column=0, sticky="ew", pady=2)
    f_out.grid_columnconfigure(0, weight=1)
    ttk.Entry(f_out, textvariable=var_out_folder).grid(row=0, column=0, sticky="ew", padx=(0,5))
    ttk.Button(f_out, text="Browse", command=lambda: choose_folder(var_out_folder)).grid(row=0, column=1)

    ttk.Label(out_frame, text="File name:").grid(row=2, column=0, sticky="w", pady=(6,0))
    ttk.Entry(out_frame, textvariable=var_out_name).grid(row=3, column=0, sticky="ew", pady=2)

    # ── Status / log ─────────────────────────────────────────────────────────
    status_box = tk.Text(root, height=7, width=70, state='disabled', wrap='word')
    status_box.grid(row=4, column=0, **pad)

    def log(msg, color=None):
        status_box.config(state='normal')
        if color:
            tag = f'color_{color}'
            status_box.tag_config(tag, foreground=color)
            status_box.insert(tk.END, msg + '\n', tag)
        else:
            status_box.insert(tk.END, msg + '\n')
        status_box.see(tk.END)
        status_box.config(state='disabled')

    def clear_log():
        status_box.config(state='normal')
        status_box.delete('1.0', tk.END)
        status_box.config(state='disabled')

    # ── Build & Save ─────────────────────────────────────────────────────────
    def build_and_save():
        clear_log()
        mtype = measurement_var.get()

        out_folder = var_out_folder.get().strip()
        out_name   = var_out_name.get().strip()
        if not out_folder or not os.path.isdir(out_folder):
            messagebox.showerror("Output error", "Please select a valid save folder.")
            return
        out_path = os.path.join(out_folder, out_name)

        try:
            if mtype == 'Multiple Spectra':
                uv_file = var_uv_mult.get().strip()
                if not uv_file:
                    raise ValueError("UV-Vis file (Multiple Spectra) is required.")
                start, step = _float(var_start.get()), _float(var_step.get())
                df = build_kinetics_df_multiple_spectra(uv_file, start, step)
                log(f"Mode         : Multiple Spectra")
                log(f"UV-Vis file  : {os.path.basename(uv_file)}")
                log(f"Times        : {df.columns[1]}, {df.columns[2] if df.shape[1] > 2 else ''}, "
                    f"... {df.columns[-1]} s ({df.shape[1] - 1} spectra)")

            else:  # Kinetics
                kin_file   = var_kinetics.get().strip()
                start_file = var_uv_start.get().strip()
                if not kin_file:
                    raise ValueError("Kinetics file is required.")
                if not start_file:
                    raise ValueError("Initial UV-Vis file is required for Kinetics mode.")
                wl = _float(var_wl.get())
                df = build_kinetics_df_kinetics(kin_file, start_file, wl)
                log(f"Mode              : Kinetics")
                log(f"Kinetics file     : {os.path.basename(kin_file)}")
                log(f"Initial UV-Vis    : {os.path.basename(start_file)}")
                log(f"Observation wl    : {wl} nm")

            df.to_csv(out_path, index=False)
            log(f"Shape        : {df.shape[0]} rows × {df.shape[1]} columns (incl. wavelength)", color='green')
            log(f"Saved to     : {out_path}", color='green')

        except Exception as e:
            log(f"Error: {e}", color='red')

    ttk.Button(
        root, text="Build & Save Kinetics DataFrame",
        command=build_and_save
    ).grid(row=5, column=0, pady=(4, 12), ipadx=10)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
