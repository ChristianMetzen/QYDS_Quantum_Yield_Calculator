"""
Kinetics DataFrame Builder + Concentration Profile
===================================================
Step 1: Build a spectral kinetics DataFrame from UV-Vis spectra + QYDS timing file.
Step 2: Calculate concentration profiles c_A(t), c_B(t) from the spectra
        using the same weighted analytical formula as the microgel notebooks
        (E-step: x_B per spectrum, weighted by |Δε|, then normalised).

Outputs
-------
  <basename>_kinetics.csv     : wavelength × time DataFrame (spectra)
  <basename>_concentrations.csv : time / s | c_<species A> / M | c_<species B> / M
"""

import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_UV_Vis_csv_multi_wl(file):
    """Read UV-Vis export where each spectrum has its own wavelength column."""
    with open(file, 'r', encoding='utf-8', errors='replace') as f:
        lines = []
        for line in f:
            if line.strip() == "":
                break
            lines.append(line)
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)), header=None,
                     na_values=["", " "], engine='python')
    spectrum_names = df.iloc[0, ::2].dropna().tolist()
    data = df.iloc[2:].copy()
    numeric_mask = pd.to_numeric(data.iloc[:, 0], errors='coerce').notna()
    data = data[numeric_mask]
    wavelengths = data.iloc[:, 0].astype(float).reset_index(drop=True)
    absorbances = data.iloc[:, 1::2].astype(float).reset_index(drop=True)
    result = pd.concat([wavelengths, absorbances], axis=1)
    result.columns = ["wavelength / nm"] + spectrum_names
    return result


def read_UV_Vis_csv_single_wl(file):
    """Read UV-Vis export where the first column is wavelength, rest are absorbances."""
    return pd.read_csv(file)


def read_QYDS_dat(file):
    """Parse a QYDS LabVIEW protocol .dat file.

    Returns DataFrame with:
      Runtime [s], Illum_Time [s], Power_mW (reference cuvette), Cell_V (solar cell voltage).
    The 'Cell (mW)' column label in the file is misleading — it stores the raw
    solar cell voltage in V; multiply by the calibration factor to get mW.
    """
    rows = []
    with open(file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\r', '')
        if line.startswith('#') or 'Number' in line or line.strip() == '':
            continue
        parts = line.split('\t')
        if len(parts) >= 7:
            try:
                rows.append({
                    'Runtime':    float(parts[3].replace(',', '.')),
                    'Illum_Time': float(parts[4].replace(',', '.')),
                    'Power_mW':   float(parts[5].replace(',', '.')),
                    'Cell_V':     float(parts[6].replace(',', '.')),
                })
            except ValueError:
                continue
    return pd.DataFrame(rows).astype(float)


def detect_spike_columns(df, threshold=0.1):
    spike_cols = []
    for col in df.columns[1:]:
        y = df[col].to_numpy(dtype=float)
        d2 = np.diff(y, n=2)
        if np.max(np.abs(d2)) > threshold:
            spike_cols.append(col)
    return spike_cols


# ─────────────────────────────────────────────────────────────────────────────
# Concentration profile
# ─────────────────────────────────────────────────────────────────────────────

def compute_concentration_profile(df_kinetics, df_eps, c_total,
                                  wl_min=330.0, wl_max=380.0):
    """
    Compute x_B(t) and x_A(t) = 1-x_B using the weighted analytical formula:
        x_B(λ) = (A(λ) - c·ε_A(λ)) / (c·(ε_B - ε_A)(λ))
    averaged over [wl_min, wl_max], weighted by |Δε|, clipped to [0,1].

    Column 1 of df_eps = species A, column 2 = species B.
    Returns DataFrame: time / s | c_<nameA> / M | c_<nameB> / M
    """
    ikw = dict(kind='linear', bounds_error=False, fill_value=np.nan)

    wl_obs = df_kinetics.iloc[:, 0].to_numpy(dtype=float)
    wl_eps = df_eps.iloc[:, 0].to_numpy(dtype=float)
    eps_A  = df_eps.iloc[:, 1].to_numpy(dtype=float)
    eps_B  = df_eps.iloc[:, 2].to_numpy(dtype=float)
    name_A = df_eps.columns[1]
    name_B = df_eps.columns[2]

    # Build common fit grid
    wl_lo  = max(wl_obs.min(), wl_eps.min(), wl_min)
    wl_hi  = min(wl_obs.max(), wl_eps.max(), wl_max)
    wl_fit = wl_obs[(wl_obs >= wl_lo) & (wl_obs <= wl_hi)]

    eps_A_fit = interp1d(wl_eps, eps_A, **ikw)(wl_fit)
    eps_B_fit = interp1d(wl_eps, eps_B, **ikw)(wl_fit)
    delta_eps = eps_B_fit - eps_A_fit
    weight    = np.abs(delta_eps)

    times   = np.array([float(c) for c in df_kinetics.columns[1:]])
    x_B_arr = []

    for col in df_kinetics.columns[1:]:
        A = interp1d(wl_obs, df_kinetics[col].to_numpy(dtype=float), **ikw)(wl_fit)
        num    = A - c_total * eps_A_fit
        den    = c_total * delta_eps
        x_wl   = np.where(np.abs(den) > 1e-12, num / den, np.nan)
        x_clip = np.clip(x_wl, 0, 1)
        fin    = np.isfinite(x_clip)
        x_B_arr.append(
            float(np.average(x_clip[fin], weights=weight[fin])) if fin.any() else np.nan
        )

    x_B = np.array(x_B_arr)
    x_A = 1.0 - x_B

    return pd.DataFrame({
        'time / s':          times,
        f'c_{name_A} / M':  x_A * c_total,
        f'c_{name_B} / M':  x_B * c_total,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():
    root = tk.Tk()
    root.title("Kinetics + Concentration Builder")
    root.resizable(False, False)
    pad = {'padx': 8, 'pady': 4}

    def _float(s):
        return float(str(s).replace(',', '.'))

    def choose_file(var, title="Open file"):
        p = filedialog.askopenfilename(title=title, filetypes=[("All files", "*.*")])
        if p:
            var.set(p)

    def choose_folder(var):
        p = filedialog.askdirectory(title="Select folder")
        if p:
            var.set(p)

    def file_row(parent, label, var, row, title="Open file"):
        ttk.Label(parent, text=label).grid(row=row*2, column=0, sticky="w", pady=(6, 0))
        f = ttk.Frame(parent)
        f.grid(row=row*2+1, column=0, sticky="ew", pady=2)
        f.grid_columnconfigure(0, weight=1)
        ttk.Entry(f, textvariable=var).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(f, text="Browse",
                   command=lambda: choose_file(var, title)).grid(row=0, column=1)

    # ── Measurement type ──────────────────────────────────────────────────────
    tf = ttk.LabelFrame(root, text="Measurement Type", padding=8)
    tf.grid(row=0, column=0, sticky="ew", **pad)
    var_mtype = tk.StringVar(value='Multiple Spectra (separate WL per spectrum)')
    ttk.Label(tf, text="Mode:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(tf, textvariable=var_mtype, state='readonly', width=40,
                 values=['Multiple Spectra (separate WL per spectrum)',
                         'Multiple Spectra (single WL column)']).grid(
        row=0, column=1, sticky="ew", padx=5)

    # ── Files ─────────────────────────────────────────────────────────────────
    ff = ttk.LabelFrame(root, text="Files", padding=8)
    ff.grid(row=1, column=0, sticky="ew", **pad)
    ff.grid_columnconfigure(0, weight=1)

    var_uv   = tk.StringVar()
    var_qyds = tk.StringVar()
    var_eps  = tk.StringVar()
    var_out  = tk.StringVar()
    var_name = tk.StringVar(value='experiment')

    file_row(ff, "UV-Vis spectra file:",              var_uv,   0, "Open UV-Vis CSV")
    file_row(ff, "QYDS .dat file (for timing):",      var_qyds, 1, "Open QYDS .dat")
    file_row(ff, "Absorption coefficients CSV:",       var_eps,  2, "Open ε file")

    ttk.Label(ff, text="Output folder:").grid(row=6, column=0, sticky="w", pady=(6, 0))
    f_out = ttk.Frame(ff)
    f_out.grid(row=7, column=0, sticky="ew", pady=2)
    f_out.grid_columnconfigure(0, weight=1)
    ttk.Entry(f_out, textvariable=var_out).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(f_out, text="Browse",
               command=lambda: choose_folder(var_out)).grid(row=0, column=1)

    ttk.Label(ff, text="Base name:").grid(row=8, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(ff, textvariable=var_name).grid(row=9, column=0, sticky="ew", pady=2)

    # ── Parameters ───────────────────────────────────────────────────────────
    pf = ttk.LabelFrame(root, text="Parameters", padding=8)
    pf.grid(row=2, column=0, sticky="ew", **pad)

    params_entries = [
        ("Concentration / µM",            tk.StringVar(value='40')),
        ("Conc. fit WL min / nm",         tk.StringVar(value='330')),
        ("Conc. fit WL max / nm",         tk.StringVar(value='380')),
        ("Spike threshold (2nd deriv.)",  tk.StringVar(value='0.1')),
    ]
    p_vars = {lbl: var for lbl, var in params_entries}
    for i, (lbl, var) in enumerate(params_entries):
        ttk.Label(pf, text=lbl + ":").grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
        ttk.Entry(pf, textvariable=var, width=12).grid(row=i, column=1, sticky="w")

    # Auto / manual timing
    ttk.Label(pf, text="Timing source:").grid(
        row=len(params_entries), column=0, sticky="w", pady=2)
    var_timing = tk.StringVar(value='QYDS file')
    ttk.Combobox(pf, textvariable=var_timing, state='readonly', width=12,
                 values=['QYDS file', 'Auto']).grid(
        row=len(params_entries), column=1, sticky="w")

    ttk.Label(pf, text="Auto start / s:").grid(
        row=len(params_entries)+1, column=0, sticky="w", pady=2)
    var_start = tk.StringVar(value='0')
    ttk.Entry(pf, textvariable=var_start, width=10).grid(
        row=len(params_entries)+1, column=1, sticky="w")

    ttk.Label(pf, text="Auto step / s:").grid(
        row=len(params_entries)+2, column=0, sticky="w", pady=2)
    var_step = tk.StringVar(value='30')
    ttk.Entry(pf, textvariable=var_step, width=10).grid(
        row=len(params_entries)+2, column=1, sticky="w")

    # ── Preview plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.tight_layout(pad=2)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=4, column=0, sticky="nsew", **pad)

    # ── Log ───────────────────────────────────────────────────────────────────
    log_box = tk.Text(root, height=7, width=80, state='disabled', wrap='word')
    log_box.grid(row=5, column=0, **pad)

    def log(msg, color=None):
        log_box.config(state='normal')
        tag = f'c_{color}' if color else None
        if tag:
            log_box.tag_config(tag, foreground=color)
            log_box.insert(tk.END, msg + '\n', tag)
        else:
            log_box.insert(tk.END, msg + '\n')
        log_box.see(tk.END)
        log_box.config(state='disabled')

    def clear_log():
        log_box.config(state='normal')
        log_box.delete('1.0', tk.END)
        log_box.config(state='disabled')

    # ── Main action ───────────────────────────────────────────────────────────
    def run():
        clear_log()
        for ax in axes:
            ax.clear()

        try:
            # ── Load UV-Vis spectra ───────────────────────────────────────────
            uv_file = var_uv.get().strip()
            if not uv_file:
                raise ValueError("UV-Vis file is required.")

            if 'single' in var_mtype.get():
                df_spectra = read_UV_Vis_csv_single_wl(uv_file)
                df_spectra.columns = ['wavelength / nm'] + list(df_spectra.columns[1:])
            else:
                df_spectra = read_UV_Vis_csv_multi_wl(uv_file)

            log(f"Loaded {df_spectra.shape[1]-1} spectra from {os.path.basename(uv_file)}")

            # ── Assign times ──────────────────────────────────────────────────
            if var_timing.get() == 'QYDS file':
                qyds_file = var_qyds.get().strip()
                if not qyds_file:
                    raise ValueError("QYDS .dat file is required for manual timing.")
                df_qyds = read_QYDS_dat(qyds_file)
                times   = df_qyds['Illum_Time'].to_numpy(dtype=float)
                log(f"Timing from QYDS file: {len(times)} time points, "
                    f"t_max = {times[-1]:.1f} s")
            else:
                n     = df_spectra.shape[1] - 1
                start = _float(var_start.get())
                step  = _float(var_step.get())
                times = np.arange(start, start + n * step, step, dtype=float)
                log(f"Auto timing: start={start} s, step={step} s, n={n}")

            n_spec = df_spectra.shape[1] - 1
            if len(times) != n_spec:
                raise ValueError(
                    f"Mismatch: {n_spec} spectra but {len(times)} time points.")

            df_spectra.columns = ['wavelength / nm'] + times.tolist()

            # ── Spike removal ─────────────────────────────────────────────────
            threshold = _float(p_vars["Spike threshold (2nd deriv.)"].get())
            spike_cols = detect_spike_columns(df_spectra, threshold)
            if spike_cols:
                log(f"Removing {len(spike_cols)} spike spectrum/a at t = "
                    + ", ".join(str(c) for c in spike_cols), color='orange')
                df_spectra = df_spectra.drop(columns=spike_cols)
            else:
                log("No spikes detected.", color='green')

            # ── Load ε file ───────────────────────────────────────────────────
            eps_file = var_eps.get().strip()
            if not eps_file:
                raise ValueError("Absorption coefficients file is required.")
            df_eps = pd.read_csv(eps_file)
            name_A = df_eps.columns[1]
            name_B = df_eps.columns[2]
            log(f"ε file: species A = '{name_A}', species B = '{name_B}'")

            # ── Concentration profile ─────────────────────────────────────────
            c_total = _float(p_vars["Concentration / µM"].get()) * 1e-6
            wl_min  = _float(p_vars["Conc. fit WL min / nm"].get())
            wl_max  = _float(p_vars["Conc. fit WL max / nm"].get())

            df_conc = compute_concentration_profile(
                df_spectra, df_eps, c_total, wl_min, wl_max)

            times_out = df_conc['time / s'].to_numpy()
            c_A = df_conc[f'c_{name_A} / M'].to_numpy()
            c_B = df_conc[f'c_{name_B} / M'].to_numpy()

            log(f"c_A: {c_A[0]*1e6:.2f} → {c_A[-1]*1e6:.2f} µM  "
                f"({name_A})")
            log(f"c_B: {c_B[0]*1e6:.2f} → {c_B[-1]*1e6:.2f} µM  "
                f"({name_B})")

            # ── Save ──────────────────────────────────────────────────────────
            out_dir  = var_out.get().strip()
            basename = var_name.get().strip()
            if not out_dir or not os.path.isdir(out_dir):
                raise ValueError("Please select a valid output folder.")

            kin_path  = os.path.join(out_dir, f"{basename}_kinetics.csv")
            conc_path = os.path.join(out_dir, f"{basename}_concentrations.csv")

            df_spectra.to_csv(kin_path,  index=False)
            df_conc.to_csv(conc_path, index=False)

            log(f"Saved: {os.path.basename(kin_path)}", color='green')
            log(f"Saved: {os.path.basename(conc_path)}", color='green')

            # ── Preview plots ─────────────────────────────────────────────────
            # Left: representative spectra
            ax = axes[0]
            wl = df_spectra.iloc[:, 0].to_numpy()
            cols = df_spectra.columns[1:]
            n_plot = min(len(cols), 12)
            idx_plot = np.round(np.linspace(0, len(cols)-1, n_plot)).astype(int)
            import matplotlib.cm as cm
            cmap = cm.viridis(np.linspace(0, 1, n_plot))
            for k, i in enumerate(idx_plot):
                ax.plot(wl, df_spectra.iloc[:, i+1].to_numpy(), color=cmap[k], lw=0.8)
            ax.set_xlabel("wavelength / nm")
            ax.set_ylabel("absorbance")
            ax.set_title("Spectra overview")
            ax.axvline(wl_min, color='grey', ls='--', lw=0.7)
            ax.axvline(wl_max, color='grey', ls='--', lw=0.7)

            # Right: concentration profiles
            ax2 = axes[1]
            ax2.plot(times_out, c_A*1e6, 'o-', ms=4, lw=1.5, label=name_A)
            ax2.plot(times_out, c_B*1e6, 's-', ms=4, lw=1.5, label=name_B)
            ax2.set_xlabel("irradiation time / s")
            ax2.set_ylabel("concentration / µM")
            ax2.set_title("Concentration profiles")
            ax2.legend(frameon=False)

            fig.tight_layout()
            canvas.draw()

        except Exception as e:
            log(f"Error: {e}", color='red')
            import traceback
            log(traceback.format_exc(), color='red')

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=3, column=0, pady=4)
    ttk.Button(btn_frame, text="Build & Save",
               command=run).pack(side='left', padx=6, ipadx=10)
    ttk.Button(btn_frame, text="Clear log",
               command=clear_log).pack(side='left', padx=6)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    run_gui()
