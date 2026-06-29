"""
QYDS Quantum Yield Calculation
================================
Implements the full spectral ODE from the ag-Riedle QYDS manual (Sec. 7, p. 7-3):

  dN_Z/dt = -Φ_{Z→E} ∫ [c_Z·ε_Z(λ) / (c_Z·ε_Z(λ)+c_E·ε_E(λ))] · [P₀·LED(λ)·λ/(h·c)] · Abs(λ) dλ
            +Φ_{E→Z} ∫ [c_E·ε_E(λ) / (c_Z·ε_Z(λ)+c_E·ε_E(λ))] · [P₀·LED(λ)·λ/(h·c)] · Abs(λ) dλ

where:
  P₀       = optical power after reference cuvette [W]  (from QYDS voltage × calibration factor)
  LED(λ)   = normalised Gaussian LED spectral distribution
  Abs(λ)   = 1 − 10^{−d·[c_Z·ε_Z(λ) + c_E·ε_E(λ)]}
  d        = cuvette path length [cm]

Inputs
------
  Concentration CSV  : time / s | c_A / M | c_B / M  (from 02_kinetics_and_concentration.py)
  Absorption ε CSV   : wavelength / nm | <name_A> | <name_B>  [L/mol/cm]
  QYDS .dat          : LabVIEW protocol with Illum_Time and Cell_V (solar cell voltage)

The QYDS file stores the solar cell voltage under "Cell (mW)".
Power = voltage × calibration_factor(λ, channel)  [mW → W ×1e-3]
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import constants
from scipy.integrate import trapezoid, solve_ivp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# SCD calibration table (from QYDS manual Table 12.4)
# ─────────────────────────────────────────────────────────────────────────────
_CAL_WL = [
    260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,
    410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,
    560,570,580,590,600,610,620,630,640,650,660
]
_CAL_HIGH = [
    10.042,9.892,9.493,8.934,8.301,7.654,7.036,6.467,5.956,5.503,
    5.105,4.758,4.454,4.189,3.958,3.756,3.578,3.423,3.285,3.164,
    3.056,2.960,2.874,2.797,2.727,2.663,2.605,2.552,2.502,2.455,
    2.410,2.367,2.325,2.284,2.244,2.203,2.162,2.121,2.080,2.038,1.995
]
_CAL_LOW = [
    0.502,0.495,0.475,0.447,0.415,0.383,0.352,0.323,0.298,0.275,
    0.255,0.238,0.223,0.209,0.198,0.188,0.179,0.171,0.164,0.158,
    0.153,0.148,0.144,0.140,0.136,0.133,0.130,0.128,0.125,0.123,
    0.120,0.118,0.116,0.114,0.112,0.110,0.108,0.106,0.104,0.102,0.100
]

def get_calibration_factor(wavelength_nm, channel):
    """Interpolate SCD calibration factor [mW/V] at the given wavelength."""
    table = _CAL_HIGH if channel == 'high' else _CAL_LOW
    return float(np.interp(wavelength_nm, _CAL_WL, table))


# ─────────────────────────────────────────────────────────────────────────────
# LED catalogue (nominal → central WL [nm], FWHM [nm])
# ─────────────────────────────────────────────────────────────────────────────
LED_CATALOGUE = {
    "278 nm  (LG Innotek LEUVA66G00HV00)":  (280,  12),
    "305 nm  (LG Innotek LEUVA66H100KU00)": (307,  13),
    "325 nm  (Dowa EOLS-325-969)":           (328,  14),
    "340 nm  (Seoul Viosys CUD4AF1B)":       (348,  10),
    "365 nm  (Seoul Viosys CUN6GB1A)":       (370,   9),
    "385 nm  (Seoul Viosys CUN8GF1A)":       (387,  10),
    "400 nm  (Luxeon LHUV-0400-0450)":       (403,  11),
    "415 nm  (Luxeon LHUV-0415-0600)":       (421,  13),
    "420 nm  (Luxeon LHUV-0420-0650)":       (426,  13),
    "425 nm  (Luxeon LHUV-0425-0600)":       (425,  13),
    "455 nm  (Osram LD-CQ7P-1U3U)":         (455,  21),
    "470 nm  (Osram LBCP7P-GYHY)":          (468,  16),
    "505 nm  (Osram LVCK7P-JYKZ)":          (507,  33),
    "528 nm  (Osram LTCP7P-KXKZ)":          (529,  36),
    "567 nm  (Luxeon Rebel ES Lime)":        (567, 103),
    "590 nm  (Osram LYCP7P-JRJT)":          (592,  17),
    "617 nm  (Osram LACP7P-KPKR)":          (620,  18),
    "625 nm  (Osram LRCP7P-JRJT)":          (630,  19),
    "645 nm  (Osram LHCP7P-2T3T)":          (656,  23),
    "660 nm  (Osram Hyper Red)":             (656,  17),
    "690 nm  (Roithner SMB1N-690D)":         (687,  22),
    "730 nm  (Osram Far Red)":               (728,  31),
    "780 nm  (Roithner SMB1N-780-02)":       (772,  28),
    "810 nm  (Roithner SMB1N-810D-02)":      (802,  27),
    "850 nm  (Osram ILH-IW01-85SL)":        (858,  30),
    "890 nm  (Roithner SMB1N-890DS)":        (880,  32),
    "940 nm  (Osram ILH-IO01-94SL)":        (940,  38),
    "980 nm  (Roithner SMB1N-980D)":         (949,  38),
    "1050 nm (Roithner SMB1N-1050-02)":      (1050, 600),
    "Custom (enter below)":                  (None, None),
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(log_path):
    logger = logging.getLogger("qyds_logger")
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        h.close(); logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# QYDS file reader
# ─────────────────────────────────────────────────────────────────────────────
def read_QYDS_dat(file):
    """Parse QYDS LabVIEW .dat file.

    Column layout (tab-separated, after the # header):
      Number | TimeStart | TimeEnd | Runtime | Illum.-Time | Power(mW) | Cell (mW) | ...

    'Cell (mW)' actually stores the solar cell voltage [V].
    'Power(mW)' is the reference-path power in mW (may be NaN in some files).
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


# ─────────────────────────────────────────────────────────────────────────────
# Full spectral ODE (Riedle manual p. 7-3)
# ─────────────────────────────────────────────────────────────────────────────

def make_led_spectrum(wl_arr, led_center, led_fwhm):
    """Normalised Gaussian LED spectrum on wl_arr."""
    sigma = led_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    S = np.exp(-0.5 * ((wl_arr - led_center) / sigma) ** 2)
    norm = trapezoid(S, wl_arr)
    return S / norm if norm > 0 else S


def build_photon_rate_spectrum(wl_arr, led_norm, P0_W):
    """
    N_ph(λ) = P₀ · LED(λ) · λ / (h·c)   [photons s⁻¹ nm⁻¹]
    where LED(λ) is already normalised (integral = 1 over nm).
    """
    hc = constants.Planck * constants.speed_of_light  # J·m
    return P0_W * led_norm * (wl_arr * 1e-9) / hc     # photons/s/nm


def spectral_ode(t, y, wl, eps_A, eps_B, N_ph, d):
    """
    Full spectral ODE: dN_A/dt and dN_B/dt in mol/s for a cuvette volume V.

    y = [N_A, N_B]  in mol  (concentration × V will be maintained by caller)
    N_ph(λ) in photons/s/nm

    Implements eq. from p. 7-3:
      dN_Z/dt = -Φ_{Z→E}·∫ frac_Z(λ)·N_ph(λ)·Abs(λ) dλ
               +Φ_{E→Z}·∫ frac_E(λ)·N_ph(λ)·Abs(λ) dλ
    where frac_Z = c_Z·ε_Z / (c_Z·ε_Z + c_E·ε_E)
    """
    phi_A, phi_B, c_A, c_B, V_L = y[0], y[1], y[2], y[3], y[4]
    # c_A, c_B stored alongside phi to allow varying concentrations
    # Actually: y = [phi_A, phi_B]  --- Φ are free params; c(t) supplied externally
    # Restructure: we integrate c_A(t), c_B(t) directly.
    raise NotImplementedError("Use build_ode_from_params instead.")


def build_rhs(wl, eps_A, eps_B, led_norm, P0_W, d, V_L, phi_A, phi_B):
    """
    Returns the RHS function dc/dt = f(t, [c_A, c_B]) for solve_ivp.
    Concentrations in mol/L, V in L, d in cm.
    P0_W: scalar power [W] (constant or mean — time-varying handled outside).
    """
    NA   = constants.Avogadro
    hc   = constants.Planck * constants.speed_of_light  # J·m
    # Pre-compute wavelength-independent part
    # N_ph(λ) [photons/s/nm] = P0_W · LED_norm(λ) · λ[m] / (h·c)
    wl_m   = wl * 1e-9
    N_ph   = P0_W * led_norm * wl_m / hc   # photons/s/nm

    def rhs(t, y):
        c_A, c_B = y
        if c_A < 0: c_A = 0.0
        if c_B < 0: c_B = 0.0
        # Absorbance spectrum
        Abs = 1.0 - 10.0 ** (-d * (c_A * eps_A + c_B * eps_B))
        denom = c_A * eps_A + c_B * eps_B
        # Avoid division by zero
        safe_denom = np.where(denom > 0, denom, 1e-30)
        frac_A = (c_A * eps_A) / safe_denom   # fraction absorbed by A
        frac_B = (c_B * eps_B) / safe_denom

        # ΔN_A/Δt [mol/s] = V[L] × Δc_A/Δt
        # dN_A_mol/dt = (-phi_A·∫frac_A·N_ph·Abs dλ + phi_B·∫frac_B·N_ph·Abs dλ) / NA
        integrand_A = frac_A * N_ph * Abs
        integrand_B = frac_B * N_ph * Abs

        rate_A_per_s = (-phi_A * trapezoid(integrand_A, wl)
                        + phi_B * trapezoid(integrand_B, wl)) / NA  # mol/s
        rate_B_per_s = -rate_A_per_s

        # Convert to concentration change [mol/L/s]
        dc_A_dt = rate_A_per_s / V_L
        dc_B_dt = rate_B_per_s / V_L
        return [dc_A_dt, dc_B_dt]

    return rhs


def simulate(times_s, c_A0, c_B0, wl, eps_A, eps_B,
             led_norm, P0_W, d_cm, V_L, phi_A, phi_B):
    """Integrate the full spectral ODE and return c_A(t), c_B(t)."""
    rhs = build_rhs(wl, eps_A, eps_B, led_norm, P0_W, d_cm, V_L, phi_A, phi_B)
    sol = solve_ivp(rhs, (times_s[0], times_s[-1]), [c_A0, c_B0],
                    t_eval=times_s, method='LSODA', rtol=1e-6, atol=1e-10)
    return sol.y[0], sol.y[1]


def fit_quantum_yields(times_s, c_A_exp, c_B_exp, c_A0, c_B0,
                       wl, eps_A, eps_B, led_norm, P0_W, d_cm, V_L,
                       print_func=print):
    """Fit Φ_A and Φ_B by least-squares on c_A(t)."""

    def model(t, phi_A, phi_B):
        try:
            cA, _ = simulate(t, c_A0, c_B0, wl, eps_A, eps_B,
                             led_norm, P0_W, d_cm, V_L, phi_A, phi_B)
            return cA
        except Exception:
            return np.full_like(t, np.nan, dtype=float)

    # Initial guess via linear pre-estimate (finite difference of ODE prefactor)
    # P(λ) = N_ph(λ) × Abs(λ) / NA
    NA = constants.Avogadro
    c_A_m = 0.5 * (c_A_exp[:-1] + c_A_exp[1:])
    c_B_m = 0.5 * (c_B_exp[:-1] + c_B_exp[1:])
    dt    = np.diff(times_s)
    dcA   = np.diff(c_A_exp) / dt   # [mol/L/s]

    pre_A_col = np.zeros(len(c_A_m))
    pre_B_col = np.zeros(len(c_A_m))
    for i in range(len(c_A_m)):
        Abs_i   = 1.0 - 10.0 ** (-d_cm * (c_A_m[i]*eps_A + c_B_m[i]*eps_B))
        denom_i = c_A_m[i]*eps_A + c_B_m[i]*eps_B
        safe    = np.where(denom_i > 0, denom_i, 1e-30)
        fA = (c_A_m[i]*eps_A) / safe
        fB = (c_B_m[i]*eps_B) / safe
        N_ph_i = P0_W * led_norm * (wl*1e-9) / (constants.Planck*constants.speed_of_light)
        rate_A  = trapezoid(fA * N_ph_i * Abs_i, wl) / (NA * V_L)
        rate_B  = trapezoid(fB * N_ph_i * Abs_i, wl) / (NA * V_L)
        pre_A_col[i] = -rate_A   # coeff of phi_A → makes c_A decrease
        pre_B_col[i] =  rate_B   # coeff of phi_B → makes c_A increase

    fin = np.isfinite(dcA) & np.isfinite(pre_A_col) & np.isfinite(pre_B_col)
    X   = np.column_stack([pre_A_col[fin], pre_B_col[fin]])
    phi_guess, *_ = np.linalg.lstsq(X, dcA[fin], rcond=None)
    phi_A_g = float(np.clip(phi_guess[0], 0, 1))
    phi_B_g = float(np.clip(phi_guess[1], 0, 1))
    print_func(f"Pre-estimate: Φ_A≈{phi_A_g:.3f}, Φ_B≈{phi_B_g:.3f}")

    popt, pcov = curve_fit(model, times_s, c_A_exp,
                           p0=[phi_A_g, phi_B_g],
                           bounds=(0, 1), maxfev=5000)
    perr = np.sqrt(np.diag(pcov))
    return popt[0], popt[1], perr[0], perr[1]


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():
    root = tk.Tk()
    root.title("QYDS Quantum Yield Calculator (Riedle method)")
    root.resizable(False, False)
    pad = {'padx': 8, 'pady': 4}

    def _float(s):
        return float(str(s).replace(',', '.'))

    def choose_file(var, title="Open file"):
        p = filedialog.askopenfilename(title=title,
                                       filetypes=[("All files", "*.*")])
        if p: var.set(p)

    def choose_folder(var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def file_row(parent, label, var, row, title="Open file"):
        ttk.Label(parent, text=label).grid(
            row=row*2, column=0, sticky="w", pady=(6, 0))
        f = ttk.Frame(parent)
        f.grid(row=row*2+1, column=0, sticky="ew", pady=2)
        f.grid_columnconfigure(0, weight=1)
        ttk.Entry(f, textvariable=var).grid(
            row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(f, text="Browse",
                   command=lambda: choose_file(var, title)).grid(row=0, column=1)

    # ── Files ─────────────────────────────────────────────────────────────────
    ff = ttk.LabelFrame(root, text="Input Files", padding=8)
    ff.grid(row=0, column=0, sticky="ew", **pad)
    ff.grid_columnconfigure(0, weight=1)

    var_conc  = tk.StringVar()
    var_eps   = tk.StringVar()
    var_qyds  = tk.StringVar()
    var_out   = tk.StringVar()
    var_name  = tk.StringVar(value='qyds_result')

    file_row(ff, "Concentration CSV  (time, c_A, c_B):",  var_conc,  0, "Open concentration CSV")
    file_row(ff, "Absorption coeff. CSV  (wl, εA, εB):", var_eps,   1, "Open ε CSV")
    file_row(ff, "QYDS .dat file:",                       var_qyds,  2, "Open QYDS .dat")

    ttk.Label(ff, text="Output folder:").grid(row=6, column=0, sticky="w", pady=(6, 0))
    f_out = ttk.Frame(ff)
    f_out.grid(row=7, column=0, sticky="ew", pady=2)
    f_out.grid_columnconfigure(0, weight=1)
    ttk.Entry(f_out, textvariable=var_out).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(f_out, text="Browse",
               command=lambda: choose_folder(var_out)).grid(row=0, column=1)
    ttk.Label(ff, text="Base name:").grid(row=8, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(ff, textvariable=var_name).grid(row=9, column=0, sticky="ew", pady=2)

    # ── LED selection ─────────────────────────────────────────────────────────
    lf = ttk.LabelFrame(root, text="LED", padding=8)
    lf.grid(row=1, column=0, sticky="ew", **pad)

    var_led_sel    = tk.StringVar(value='Custom (enter below)')
    var_led_center = tk.StringVar(value='365')
    var_led_fwhm   = tk.StringVar(value='9')

    ttk.Label(lf, text="LED:").grid(row=0, column=0, sticky="w")
    led_cb = ttk.Combobox(lf, textvariable=var_led_sel,
                          values=list(LED_CATALOGUE.keys()),
                          state='readonly', width=40)
    led_cb.grid(row=0, column=1, sticky="ew", padx=5)

    ttk.Label(lf, text="Central WL / nm:").grid(row=1, column=0, sticky="w", pady=2)
    led_center_e = ttk.Entry(lf, textvariable=var_led_center, width=10)
    led_center_e.grid(row=1, column=1, sticky="w", padx=5)

    ttk.Label(lf, text="FWHM / nm:").grid(row=2, column=0, sticky="w", pady=2)
    led_fwhm_e = ttk.Entry(lf, textvariable=var_led_fwhm, width=10)
    led_fwhm_e.grid(row=2, column=1, sticky="w", padx=5)

    def on_led_select(event=None):
        name = var_led_sel.get()
        center, fwhm = LED_CATALOGUE.get(name, (None, None))
        if center is not None:
            var_led_center.set(str(center))
            var_led_fwhm.set(str(fwhm))
            state = 'disabled'
        else:
            state = 'normal'
        led_center_e.config(state=state)
        led_fwhm_e.config(state=state)
    led_cb.bind('<<ComboboxSelected>>', on_led_select)

    # ── Parameters ───────────────────────────────────────────────────────────
    pf = ttk.LabelFrame(root, text="Parameters", padding=8)
    pf.grid(row=2, column=0, sticky="ew", **pad)

    param_list = [
        ("Voltage after reference / V",  tk.StringVar(value='7.46')),
        ("Channel (high / low)",         tk.StringVar(value='low')),
        ("Cuvette path length / cm",     tk.StringVar(value='1')),
        ("Volume / mL",                  tk.StringVar(value='3')),
    ]
    p_vars = {lbl: var for lbl, var in param_list}
    for i, (lbl, var) in enumerate(param_list):
        ttk.Label(pf, text=lbl + ":").grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
        if lbl == "Channel (high / low)":
            ttk.Combobox(pf, textvariable=var, values=['low', 'high'],
                         state='readonly', width=8).grid(row=i, column=1, sticky="w")
        else:
            ttk.Entry(pf, textvariable=var, width=12).grid(row=i, column=1, sticky="w")

    # ── Output display ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.tight_layout(pad=2)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=4, column=0, sticky="nsew", **pad)

    out_box = tk.Text(root, height=10, width=80, state='disabled', wrap='word')
    out_box.grid(row=5, column=0, **pad)

    def log(msg, color=None):
        out_box.config(state='normal')
        tag = f'c_{color}' if color else None
        if tag:
            out_box.tag_config(tag, foreground=color)
            out_box.insert(tk.END, msg + '\n', tag)
        else:
            out_box.insert(tk.END, msg + '\n')
        out_box.see(tk.END)
        out_box.config(state='disabled')
        root.update_idletasks()

    def clear_log():
        out_box.config(state='normal')
        out_box.delete('1.0', tk.END)
        out_box.config(state='disabled')

    # ── Run ───────────────────────────────────────────────────────────────────
    def run():
        clear_log()
        for ax in axes:
            ax.clear()

        try:
            # ── Load inputs ──────────────────────────────────────────────────
            df_conc = pd.read_csv(var_conc.get().strip())
            df_eps  = pd.read_csv(var_eps.get().strip())

            times_s = df_conc.iloc[:, 0].to_numpy(dtype=float)
            c_A_exp = df_conc.iloc[:, 1].to_numpy(dtype=float)
            c_B_exp = df_conc.iloc[:, 2].to_numpy(dtype=float)
            name_A  = df_conc.columns[1].replace('c_', '').replace(' / M', '')
            name_B  = df_conc.columns[2].replace('c_', '').replace(' / M', '')

            wl_eps  = df_eps.iloc[:, 0].to_numpy(dtype=float)
            eps_A   = df_eps.iloc[:, 1].to_numpy(dtype=float)
            eps_B   = df_eps.iloc[:, 2].to_numpy(dtype=float)

            log(f"Loaded: {len(times_s)} time points, t_max={times_s[-1]:.1f} s")
            log(f"Species A: {name_A}  Species B: {name_B}")
            log(f"c_A: {c_A_exp[0]*1e6:.2f} → {c_A_exp[-1]*1e6:.2f} µM")
            log(f"c_B: {c_B_exp[0]*1e6:.2f} → {c_B_exp[-1]*1e6:.2f} µM")

            # ── Power from QYDS file ─────────────────────────────────────────
            qyds_file = var_qyds.get().strip()
            channel   = p_vars["Channel (high / low)"].get()
            led_center = _float(var_led_center.get())
            led_fwhm   = _float(var_led_fwhm.get())

            cf_mW_per_V = get_calibration_factor(led_center, channel)
            log(f"Calibration factor: {cf_mW_per_V:.3f} mW/V  "
                f"({channel}, λ_center={led_center:.0f} nm)")

            if qyds_file:
                df_qyds = read_QYDS_dat(qyds_file)
                # Mean cell voltage over the run (excluding t=0 which is 0)
                cell_V = df_qyds['Cell_V'].to_numpy()
                cell_V = cell_V[cell_V > 0]
                V_mean = float(np.nanmean(cell_V)) if len(cell_V) > 0 else 0.0
                P0_mW  = V_mean * cf_mW_per_V
                log(f"QYDS: mean cell voltage = {V_mean:.4f} V  →  "
                    f"P₀ = {P0_mW:.4f} mW")
            else:
                # Fallback: use manual voltage entry
                V_manual = _float(p_vars["Voltage after reference / V"].get())
                P0_mW    = V_manual * cf_mW_per_V
                log(f"Manual voltage: {V_manual:.4f} V  →  P₀ = {P0_mW:.4f} mW")

            P0_W  = P0_mW * 1e-3   # → Watts
            d_cm  = _float(p_vars["Cuvette path length / cm"].get())
            V_L   = _float(p_vars["Volume / mL"].get()) * 1e-3

            log(f"P₀ = {P0_W*1e3:.4f} mW,  d = {d_cm} cm,  V = {V_L*1e3:.1f} mL")

            # ── LED spectrum ─────────────────────────────────────────────────
            # Use ε wavelength grid (already covers absorption spectrum)
            order = np.argsort(wl_eps)
            wl     = wl_eps[order]
            eps_A  = eps_A[order]
            eps_B  = eps_B[order]

            led_norm = make_led_spectrum(wl, led_center, led_fwhm)
            log(f"LED: centre={led_center:.0f} nm, FWHM={led_fwhm:.0f} nm  "
                f"({var_led_sel.get()})")

            # ── Fit ──────────────────────────────────────────────────────────
            log("Fitting quantum yields ...")
            phi_A, phi_B, err_A, err_B = fit_quantum_yields(
                times_s, c_A_exp, c_B_exp,
                c_A_exp[0], c_B_exp[0],
                wl, eps_A, eps_B,
                led_norm, P0_W, d_cm, V_L,
                print_func=log)

            log(f"\nΦ({name_A}→{name_B}) = {phi_A:.4e}  ({phi_A*100:.2f} %)  "
                f"± {err_A:.2e}", color='blue')
            log(f"Φ({name_B}→{name_A}) = {phi_B:.4e}  ({phi_B*100:.2f} %)  "
                f"± {err_B:.2e}", color='blue')

            # ── PSS photon dose ──────────────────────────────────────────────
            PSS_TOL  = 0.01
            t_long   = np.linspace(times_s[0], times_s[-1]*20, 10000)
            rhs_long = build_rhs(wl, eps_A, eps_B, led_norm, P0_W, d_cm, V_L,
                                 phi_A, phi_B)
            sol_long = solve_ivp(rhs_long, (t_long[0], t_long[-1]),
                                 [c_A_exp[0], c_B_exp[0]],
                                 t_eval=t_long, method='LSODA')
            cA_long  = sol_long.y[0]
            cA_pss   = cA_long[-1]
            change   = abs(cA_pss - c_A_exp[0])
            if change > 0:
                within  = np.abs(cA_long - cA_pss) <= PSS_TOL * change
                idx_pss = int(np.argmax(within)) if within.any() else len(t_long)-1
            else:
                idx_pss = 0
            t_pss = float(t_long[idx_pss])

            Q_inc = P0_W * t_pss / (constants.Planck * constants.speed_of_light /
                                    (led_center * 1e-9)) / constants.Avogadro
            Abs_long = 1.0 - 10.0 ** (-d_cm * (sol_long.y[0]*eps_A[:, None]
                                                 if False else
                                                 np.outer(eps_A, sol_long.y[0]).sum(0)))
            # Simplified Q_inc/Q_abs using average absorbance
            A_mean_long = d_cm * ((sol_long.y[0]*np.interp(led_center, wl, eps_A)) +
                                  (sol_long.y[1]*np.interp(led_center, wl, eps_B)))
            Fabs_long   = 1.0 - 10.0**(-A_mean_long)
            dt_long     = np.diff(t_long)
            Q_abs_mol   = float(np.trapz(P0_W * Fabs_long[:idx_pss+1], t_long[:idx_pss+1])
                                / (constants.Planck * constants.speed_of_light
                                   / (led_center * 1e-9)) / constants.Avogadro)
            Q_inc_mol   = P0_W * t_pss / (constants.Planck * constants.speed_of_light
                                           / (led_center * 1e-9)) / constants.Avogadro
            log(f"\nPSS at t ≈ {t_pss:.1f} s  (within {PSS_TOL*100:.0f}% of asymptote)")
            log(f"Q_inc to PSS : {Q_inc_mol:.3e} mol photons")
            log(f"Q_abs to PSS : {Q_abs_mol:.3e} mol photons")

            # ── Save log ─────────────────────────────────────────────────────
            out_dir  = var_out.get().strip()
            basename = var_name.get().strip()
            if out_dir and os.path.isdir(out_dir):
                logger = setup_logger(os.path.join(out_dir, f"{basename}.log"))
                logger.info(f"LED: {var_led_sel.get()}, centre={led_center:.0f} nm, "
                            f"FWHM={led_fwhm:.0f} nm")
                logger.info(f"P0={P0_W*1e3:.4f} mW, d={d_cm} cm, V={V_L*1e3:.1f} mL")
                logger.info(f"QY {name_A}->{name_B} = {phi_A:.4e} ({phi_A*100:.2f} %)")
                logger.info(f"QY {name_B}->{name_A} = {phi_B:.4e} ({phi_B*100:.2f} %)")
                logger.info(f"PSS at t={t_pss:.2f} s; "
                            f"Q_inc={Q_inc_mol:.4e} mol, Q_abs={Q_abs_mol:.4e} mol")
                log(f"Log saved to {basename}.log", color='green')

            # ── Plot ─────────────────────────────────────────────────────────
            # Simulate fit curve on fine grid
            t_fine = np.linspace(times_s[0], times_s[-1], 400)
            cA_fit, cB_fit = simulate(t_fine, c_A_exp[0], c_B_exp[0],
                                      wl, eps_A, eps_B, led_norm, P0_W,
                                      d_cm, V_L, phi_A, phi_B)

            ax0, ax1 = axes[0], axes[1]
            ax0.plot(times_s, c_A_exp*1e6, 'o', ms=5, label=f'{name_A} (data)')
            ax0.plot(times_s, c_B_exp*1e6, 's', ms=5, label=f'{name_B} (data)')
            ax0.plot(t_fine, cA_fit*1e6, '-', lw=2, label=f'{name_A} (fit)')
            ax0.plot(t_fine, cB_fit*1e6, '-', lw=2, label=f'{name_B} (fit)')
            ax0.set_xlabel("irradiation time / s")
            ax0.set_ylabel("concentration / µM")
            ax0.set_title(f"Φ({name_A}→{name_B})={phi_A*100:.1f}%   "
                          f"Φ({name_B}→{name_A})={phi_B*100:.1f}%")
            ax0.legend(frameon=False, fontsize=8)

            ax1.plot(wl, led_norm / led_norm.max(), color='grey', lw=1.5, label='LED')
            ax1b = ax1.twinx()
            ax1b.plot(wl, eps_A, label=f'ε({name_A})', lw=1.5)
            ax1b.plot(wl, eps_B, label=f'ε({name_B})', lw=1.5, ls='--')
            ax1.set_xlabel("wavelength / nm")
            ax1.set_ylabel("LED (norm.)", color='grey')
            ax1b.set_ylabel("ε / L mol⁻¹ cm⁻¹")
            ax1b.legend(frameon=False, fontsize=8)
            ax1.set_title("LED spectrum & absorption coefficients")

            fig.tight_layout()
            canvas.draw()

        except Exception as e:
            import traceback
            log(f"Error: {e}", color='red')
            log(traceback.format_exc(), color='red')

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=3, column=0, pady=4)
    ttk.Button(btn_frame, text="Run QYDS Fit", command=run).pack(
        side='left', padx=6, ipadx=14)
    ttk.Button(btn_frame, text="Clear", command=clear_log).pack(
        side='left', padx=6)

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
