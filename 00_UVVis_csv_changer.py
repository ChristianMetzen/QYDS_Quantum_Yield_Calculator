import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

def read_UV_Vis_csv(file=None):
    
    if not file:
        return

    with open(file, 'r', encoding='utf-8', errors='replace') as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped == "":
                break
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
    data = data[numeric_mask].reset_index(drop=True)

    # Alle Wellenlängen- und Absorbanzspalten paarweise zusammenbauen
    frames = []
    for i, name in enumerate(spectrum_names):
        wl_col = i * 2
        abs_col = i * 2 + 1
        pair = data.iloc[:, [wl_col, abs_col]].astype(float)
        pair.columns = [f"Wavelength {name} (nm)", name]
        frames.append(pair)

    result = pd.concat(frames, axis=1)

    return result

def choose_file(tree):
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    df = read_UV_Vis_csv(file_path)
    if df is None:
        return

    update_table(df, tree)
    return df

def save_csv(tree):
    cols = tree["columns"]
    if len(cols) == 0:
        print("No data to save.")
        return

    # Create dataframe from displayed data
    displayed_data = []
    for item in tree.get_children():
        displayed_data.append(tree.item(item)["values"])

    df_to_save = pd.DataFrame(displayed_data, columns=cols)

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    df_to_save.to_csv(file_path, index=False)
    print(f"CSV saved to: {file_path}")

def update_table(df, tree):
    tree.delete(*tree.get_children())
    tree["columns"] = list(df.columns)
    for i, col in enumerate(df.columns):
        tree.heading(col, text=f"{i}: {col}")

        # Auto-width calculation
        max_width = max([len(str(col))] + [len(str(val)) for val in df[col]]) * 8
        tree.column(col, width=max_width, minwidth=50, stretch=False)

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

def apply_filter(df_original, tree, first_col_entry, last_col_entry):
    try:
        fc_str = first_col_entry.get()
        lc_str = last_col_entry.get()
        if fc_str == "" or lc_str == "":
            print("Please enter first and last column indices.")
            return
        fc = int(fc_str)
        lc = int(lc_str)
        df_filtered = df_original.iloc[:, fc:lc+1]
        update_table(df_filtered, tree)
        return df_filtered
    except Exception as e:
        print("Filter error:", e)

def main():
    root = tk.Tk()
    root.title("UV Vis CSV Editor")
    root.geometry("900x600")
    root.resizable(False, False)

    # Placeholder for the current dataframe
    df_original = pd.DataFrame()

    # -------------------------
    # Table frame
    # -------------------------
    table_frame = ttk.Frame(root)
    table_frame.pack(fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(table_frame, show="headings")

    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)

    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    # -------------------------
    # Controls frame
    # -------------------------
    controls = ttk.Frame(root)
    controls.pack(fill=tk.X, pady=10)

    # Choose file button
    def choose_file_callback():
        nonlocal df_original
        df_original = choose_file(tree)

    ttk.Button(controls, text="Choose CSV File", command=choose_file_callback).grid(row=0, column=0, padx=5)

    # First/Last column inputs
    ttk.Label(controls, text="First col:").grid(row=0, column=1)
    first_col_entry = ttk.Entry(controls, width=6)
    first_col_entry.grid(row=0, column=2)

    ttk.Label(controls, text="Last col:").grid(row=0, column=3)
    last_col_entry = ttk.Entry(controls, width=6)
    last_col_entry.grid(row=0, column=4)

    # Apply filter button
    def apply_filter_callback():
        nonlocal df_original
        df_original = apply_filter(df_original, tree, first_col_entry, last_col_entry)

    ttk.Button(controls, text="Apply Filter", command=apply_filter_callback).grid(row=0, column=5, padx=5)

    # Save CSV button
    ttk.Button(controls, text="Save CSV", command=lambda: save_csv(tree)).grid(row=0, column=6, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()