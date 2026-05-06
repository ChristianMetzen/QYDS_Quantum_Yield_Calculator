import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_raw(file_path):
    """
    Reads the UV-Vis CSV, stopping at the first line that contains only
    commas/whitespace (instrument metadata separator). Returns:
        df          – DataFrame of spectral data only (strings)
        line_ending – original line ending bytes (e.g. b'\r\r\n')
        n_raw_cols  – number of comma-separated fields per data line
                      (needed to reconstruct trailing commas on save)
    """
    with open(file_path, 'rb') as f:
        raw = f.read()

    # Detect line ending: instrument files often use \r\r\n
    if b'\r\r\n' in raw:
        line_ending = b'\r\r\n'
    elif b'\r\n' in raw:
        line_ending = b'\r\n'
    else:
        line_ending = b'\n'

    lines_bytes = raw.split(line_ending)

    # Keep only lines up to (not including) the first all-comma/empty line
    data_lines = []
    for lb in lines_bytes:
        stripped = lb.replace(b',', b'').strip()
        if stripped == b'':
            break
        data_lines.append(lb.decode('utf-8', errors='replace'))

    # Count columns from first line (includes trailing comma → one extra empty field)
    n_raw_cols = len(data_lines[0].split(',')) if data_lines else 0

    from io import StringIO
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        header=None,
        dtype=str,
        keep_default_na=False,
    )

    return df, line_ending, n_raw_cols


def spectrum_names_from_df(df):
    """
    Row 0 contains spectrum names in every even column (odd columns empty).
    Returns list of (col_index, name) for each spectrum pair.
    """
    names = []
    for col in range(0, df.shape[1], 2):
        name = df.iloc[0, col].strip()
        if name:
            names.append((col, name))
    return names


def delete_spectra_from_df(df, cols_to_delete_start):
    """
    Given a list of starting column indices (even cols) to delete,
    removes each pair (col, col+1) from the DataFrame.
    Returns a new DataFrame.
    """
    drop_cols = []
    for c in cols_to_delete_start:
        drop_cols.extend([c, c + 1])
    keep_cols = [c for c in range(df.shape[1]) if c not in drop_cols]
    result = df.iloc[:, keep_cols].reset_index(drop=True)
    result.columns = range(result.shape[1])
    return result


def save_df_as_csv(df, file_path, line_ending=b'\r\r\n', n_raw_cols=None):
    out_lines = []
    for _, row in df.iterrows():
        fields = list(row.astype(str))
        line_str = ",".join(fields)
        out_lines.append(line_str.encode('utf-8') + line_ending)

    with open(file_path, 'wb') as f:
        f.writelines(out_lines)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UV-Vis Column Editor")
        self.root.geometry("1100x650")

        self.df = None          # current DataFrame (strings)
        self.file_path = None   # source file path
        self.check_vars = {}    # spectrum name -> BooleanVar (checked = delete)
        self.line_ending = b'\r\r\n'
        self.n_raw_cols = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # --- Top button bar ---
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=8, pady=6)

        ttk.Button(bar, text="📂  Open CSV", command=self.open_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="🗑  Delete selected columns", command=self.delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="💾  Save as …", command=self.save_file).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="No file loaded.")
        ttk.Label(bar, textvariable=self.status_var, foreground="gray").pack(side=tk.LEFT, padx=12)

        # --- Checkbox area for spectra ---
        cb_outer = ttk.LabelFrame(self.root, text="Spectra  (check to DELETE)")
        cb_outer.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.cb_canvas = tk.Canvas(cb_outer, height=52, highlightthickness=0)
        self.cb_scroll = ttk.Scrollbar(cb_outer, orient="horizontal", command=self.cb_canvas.xview)
        self.cb_canvas.configure(xscrollcommand=self.cb_scroll.set)
        self.cb_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.cb_canvas.pack(fill=tk.X, expand=True)

        self.cb_frame = ttk.Frame(self.cb_canvas)
        self.cb_canvas.create_window((0, 0), window=self.cb_frame, anchor="nw")
        self.cb_frame.bind("<Configure>",
                           lambda e: self.cb_canvas.configure(
                               scrollregion=self.cb_canvas.bbox("all")))

        # --- Table ---
        table_frame = ttk.Frame(self.root)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.tree = ttk.Treeview(table_frame, show="headings")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return
        self.file_path = path
        self.df, self.line_ending, self.n_raw_cols = read_raw(path)
        self._refresh_checkboxes()
        self._refresh_table()
        self.status_var.set(f"Loaded: {path.split('/')[-1]}  "
                            f"({self.df.shape[1]} columns, {self.df.shape[0]} rows)")

    def delete_selected(self):
        if self.df is None:
            return
        spectra = spectrum_names_from_df(self.df)
        to_delete = [col for col, name in spectra
                     if self.check_vars.get(name, tk.BooleanVar()).get()]
        if not to_delete:
            messagebox.showinfo("Nothing selected", "Check at least one spectrum to delete.")
            return
        names = [name for col, name in spectra if col in to_delete]
        if not messagebox.askyesno("Confirm delete",
                                   f"Delete these spectra?\n\n" + "\n".join(names)):
            return
        self.df = delete_spectra_from_df(self.df, to_delete)
        self._refresh_checkboxes()
        self._refresh_table()
        self.status_var.set(f"Deleted {len(names)} spectrum/spectra. "
                            f"Now {self.df.shape[1]} columns.")

    def save_file(self):
        if self.df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=self.file_path.split("/")[-1] if self.file_path else "output.csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return
        save_df_as_csv(self.df, path, self.line_ending, self.n_raw_cols)
        self.status_var.set(f"Saved to: {path.split('/')[-1]}")

    # -----------------------------------------------------------------------
    # UI refresh helpers
    # -----------------------------------------------------------------------

    def _refresh_checkboxes(self):
        for w in self.cb_frame.winfo_children():
            w.destroy()
        self.check_vars.clear()

        if self.df is None:
            return

        spectra = spectrum_names_from_df(self.df)
        for col, name in spectra:
            var = tk.BooleanVar(value=False)
            self.check_vars[name] = var
            ttk.Checkbutton(self.cb_frame, text=name, variable=var).pack(
                side=tk.LEFT, padx=6, pady=4)

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        if self.df is None:
            return

        cols = [str(i) for i in range(self.df.shape[1])]
        self.tree["columns"] = cols
        for i, col in enumerate(cols):
            header_text = self.df.iloc[0, i] if self.df.shape[0] > 0 else col
            self.tree.heading(col, text=f"{i}: {header_text}")
            max_w = max(
                [len(str(header_text))] +
                [len(str(self.df.iloc[r, i])) for r in range(min(5, self.df.shape[0]))]
            ) * 8
            self.tree.column(col, width=max(max_w, 60), minwidth=40, stretch=False)

        # Show first 200 rows (performance)
        for r in range(min(200, self.df.shape[0])):
            self.tree.insert("", "end", values=list(self.df.iloc[r]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()