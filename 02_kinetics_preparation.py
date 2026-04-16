import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

# ─────────────────────────────────────────────────────────────────────────────
# Data processing functions
# ─────────────────────────────────────────────────────────────────────────────

def read_UV_Vis_csv(file=None):
    if not file:
        return

    # Lese roh ein und schneide bei erster Leerzeile ab
    with open(file, 'r', encoding='utf-8', errors='replace') as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped == "":
                break  # Erste Leerzeile = Ende der Spektrendaten
            lines.append(line)

    from io import StringIO
    df = pd.read_csv(
        StringIO("".join(lines)),
        header=None,
        na_values=["", " "],
        engine='python',
    )

    # Spektren-Namen aus Zeile 0 (jede gerade Spalte)
    spectrum_names = df.iloc[0, ::2].dropna().tolist()

    # Ab Zeile 2, nur numerische Zeilen (Sicherheitsnetz)
    data = df.iloc[2:].copy()
    numeric_mask = pd.to_numeric(data.iloc[:, 0], errors='coerce').notna()
    data = data[numeric_mask]

    # Wellenlänge + Absorbanzspalten zusammenbauen
    wavelengths = data.iloc[:, 0].astype(float).reset_index(drop=True)
    absorbances = data.iloc[:, 1::2].astype(float).reset_index(drop=True)

    result = pd.concat([wavelengths, absorbances], axis=1)
    result.columns = ["Wavelength (nm)"] + spectrum_names

    return result


def read_QYDS_dat(file):
    data = {'Runtime': [], 'Illum Time': [], 'Cell Value': []}
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or line.strip().startswith('Number'):
            continue
        parts = line.split('\t')
        if len(parts) >= 7:
            try:
                data['Runtime'].append(float(parts[3].replace(',', '.')))
                data['Illum Time'].append(float(parts[4].replace(',', '.')))
                data['Cell Value'].append(float(parts[6].replace(',', '.')))
            except ValueError:
                continue
    return pd.DataFrame(data).astype(float)


def build_kinetics_df_multiple_spectra(uv_vis_file, qyds_file, mode, start=None, step=None):
    df_spectra = read_UV_Vis_csv(uv_vis_file)

    if mode == 'manual':
        df_qyds = read_QYDS_dat(qyds_file)
        times = df_qyds['Illum Time'].astype(float).to_numpy()
    else:
        n = df_spectra.shape[1] - 1
        times = np.arange(start, start + n * step, step, dtype=float)

    n_spectra = df_spectra.shape[1] - 1
    n_times   = len(times)

    if n_times != n_spectra:
        raise ValueError(
            f"Mismatch: {n_spectra} spectra in UV-Vis file but {n_times} time points "
            f"(from QYDS file)."
        )

    df_spectra.columns = ['wavelength / nm'] + times[:n_spectra].tolist()
    return df_spectra


def build_kinetics_df_kinetics(kinetics_file, uv_vis_start_file, irradiation_wavelength):
    """Build single-row kinetics DataFrame from a kinetics trace + initial UV-Vis."""

    df         = read_UV_Vis_csv(kinetics_file).T
    absorbance = df.iloc[1, :-1].astype(float).to_numpy().copy()  # .copy() avoids read-only error on transposed array
    times      = df.iloc[0, :-1].astype(float)

    # Baseline correction from initial UV-Vis
    df_init = read_UV_Vis_csv(uv_vis_start_file)
    wl_idx  = (df_init.iloc[:, 0] - irradiation_wavelength).abs().idxmin()
    A_ini   = df_init.iloc[wl_idx, 1]
    dA      = A_ini - absorbance[0]
    absorbance += dA

    df_kinetics = pd.DataFrame(data=[absorbance], columns=times.astype(str))
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
        ttk.Label(parent, text=label).grid(row=row*2,   column=0, sticky="w", pady=(6,0))
        f = ttk.Frame(parent)
        f.grid(row=row*2+1, column=0, sticky="ew", pady=2)
        f.grid_columnconfigure(0, weight=1)
        ttk.Entry(f, textvariable=var).grid(row=0, column=0, sticky="ew", padx=(0,5))
        ttk.Button(f, text="Browse", command=lambda: choose_file(var)).grid(row=0, column=1)

    var_uv_mult   = tk.StringVar()
    var_qyds      = tk.StringVar()
    var_kinetics  = tk.StringVar()
    var_uv_start  = tk.StringVar()

    file_row(file_frame, "UV-Vis file (Multiple Spectra)",             var_uv_mult,  0)
    file_row(file_frame, "QYDS .dat file (Multiple Spectra, manual)",  var_qyds,     1)
    file_row(file_frame, "Kinetics file (Kinetics mode)",              var_kinetics, 2)
    file_row(file_frame, "Initial UV-Vis file (Kinetics mode)",        var_uv_start, 3)

    # ── Parameters ───────────────────────────────────────────────────────────
    param_frame = ttk.LabelFrame(root, text="Parameters", padding=8)
    param_frame.grid(row=2, column=0, sticky="ew", **pad)
    param_frame.grid_columnconfigure(1, weight=1)

    var_wl    = tk.StringVar(value='365')
    var_mode  = tk.StringVar(value='manual')
    var_start = tk.StringVar(value='0')
    var_step  = tk.StringVar(value='30')

    ttk.Label(param_frame, text="Irradiation wavelength / nm:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(param_frame, textvariable=var_wl, width=12).grid(row=0, column=1, sticky="w", padx=5)

    ttk.Label(param_frame, text="Manual / Auto (Multiple Spectra):").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Combobox(
        param_frame, textvariable=var_mode,
        values=['manual', 'auto'], state='readonly', width=10
    ).grid(row=1, column=1, sticky="w", padx=5)

    # Auto time entries — shown/hidden based on mode
    auto_labels = ["Start / s", "Step / s"]
    auto_vars   = [var_start, var_step]
    auto_rows   = []
    for i, (lbl, var) in enumerate(zip(auto_labels, auto_vars)):
        l = ttk.Label(param_frame, text=lbl + ":")
        e = ttk.Entry(param_frame, textvariable=var, width=10)
        auto_rows.append((l, e, i + 2))

    def update_auto_rows(*_):
        if var_mode.get() == 'auto':
            for l, e, row in auto_rows:
                l.grid(row=row, column=0, sticky="w", pady=2)
                e.grid(row=row, column=1, sticky="w", padx=5)
        else:
            for l, e, _ in auto_rows:
                l.grid_remove()
                e.grid_remove()

    var_mode.trace_add('write', update_auto_rows)
    update_auto_rows()  # initialize

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
    status_box = tk.Text(root, height=6, width=70, state='disabled', wrap='word')
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

        try:
            wl = float(var_wl.get())
        except ValueError:
            messagebox.showerror("Input error", "Irradiation wavelength must be a number.")
            return

        out_folder = var_out_folder.get().strip()
        out_name   = var_out_name.get().strip()
        if not out_folder or not os.path.isdir(out_folder):
            messagebox.showerror("Output error", "Please select a valid save folder.")
            return
        out_path = os.path.join(out_folder, out_name)

        try:
            if mtype == 'Multiple Spectra':
                uv_file   = var_uv_mult.get().strip()
                qyds_file = var_qyds.get().strip()
                mode      = var_mode.get()

                if not uv_file:
                    raise ValueError("UV-Vis file (Multiple Spectra) is required.")

                if mode == 'manual':
                    if not qyds_file:
                        raise ValueError("QYDS .dat file is required in manual mode.")
                    log(f"Mode         : Multiple Spectra / manual")
                    log(f"UV-Vis file  : {os.path.basename(uv_file)}")
                    log(f"QYDS file    : {os.path.basename(qyds_file)}")
                    df = build_kinetics_df_multiple_spectra(uv_file, qyds_file, mode='manual')

                else:
                    start = int(var_start.get())
                    step  = int(var_step.get())
                    log(f"Mode         : Multiple Spectra / auto (start {start} s, step {step} s)")
                    log(f"UV-Vis file  : {os.path.basename(uv_file)}")
                    df = build_kinetics_df_multiple_spectra(
                        uv_file, None, mode='auto',
                        start=start, step=step
                    )

            else:  # Kinetics
                kin_file   = var_kinetics.get().strip()
                start_file = var_uv_start.get().strip()
                if not kin_file:
                    raise ValueError("Kinetics file is required.")
                if not start_file:
                    raise ValueError("Initial UV-Vis file is required for Kinetics mode.")

                log(f"Mode              : Kinetics")
                log(f"Kinetics file     : {os.path.basename(kin_file)}")
                log(f"Initial UV-Vis    : {os.path.basename(start_file)}")
                log(f"Irradiation wl    : {wl} nm")
                df = build_kinetics_df_kinetics(kin_file, start_file, wl)

            df.to_csv(out_path, index=False)
            log(f"Shape            : {df.shape[0]} rows × {df.shape[1]} columns (incl. wavelength)", color='green')
            log(f"Saved to         : {out_path}", color='green')

        except Exception as e:
            log(f"Error: {e}", color='red')

    ttk.Button(
        root, text="Build & Save Kinetics DataFrame",
        command=build_and_save
    ).grid(row=5, column=0, pady=(4, 12), ipadx=10)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
