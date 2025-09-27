#!/usr/bin/env python3
"""
GUI for forecasting animal bite cases using SARIMA and Prophet.

- Choose CSV, level (all/province/municipality/barangay), filters, model, horizon
- Run forecast and view a plot
- Saves outputs (CSVs + plot) using the same logic as forecast.py

Requires: pandas, numpy, statsmodels, matplotlib; prophet (optional)
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Set dark matplotlib theme
plt.style.use('dark_background')

# Import logic from forecast.py living in the same directory
try:
    import forecast as fcore
    from forecast import (
        read_csv_safely,
        prepare_series,
        infer_seasonal_period_and_freq,
        forecast_with_sarima,
        forecast_with_prophet,
        save_outputs,
        PROPHET_AVAILABLE,
    )
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        "Could not import forecast.py. Ensure forecast.py is in the same folder as this GUI script.\n" + str(e)
    )


def find_col_case_insensitive(df: pd.DataFrame, name: str) -> str:
    mapping = {c.lower(): c for c in df.columns}
    return mapping.get(name.lower(), name)


class ForecastApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Animal Bites Forecast (SARIMA/Prophet)")
        self.geometry("980x680")
        
        # Apply dark theme
        self._apply_dark_theme()

        # Data holders
        self.df: pd.DataFrame | None = None
        self.prov_col = "PROV_CODE"
        self.mun_col = "MUN_CODE"
        self.bgy_col = "BGY_CODE"
        self.date_col = "DATE"
        self.male_col = "RAB_ANIMBITE_M"
        self.female_col = "RAB_ANIMBITE_F"

        self.last_output_dir: str | None = None

        # Top-level frames
        self._build_file_frame()
        self._build_options_frame()
        self._build_buttons_frame()
        self._build_plot_and_preview()

        # Try to auto-load default CSV in same folder
        default_csv = os.path.join(os.path.dirname(__file__), "Animal Bites Cases.csv")
        if os.path.exists(default_csv):
            self.csv_path_var.set(default_csv)
            self.load_csv()
    
    def _apply_dark_theme(self):
        """Apply dark theme colors to the application"""
        # Dark theme colors
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        select_bg = "#404040"
        entry_bg = "#3c3c3c"
        button_bg = "#404040"
        
        # Configure root window
        self.configure(bg=bg_color)
        
        # Configure default ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure ttk widget styles for dark theme
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TLabelFrame', background=bg_color, foreground=fg_color)
        style.configure('TLabelFrame.Label', background=bg_color, foreground=fg_color)
        style.configure('TFrame', background=bg_color)
        style.configure('TEntry', fieldbackground=entry_bg, foreground=fg_color, borderwidth=1)
        style.configure('TCombobox', fieldbackground=entry_bg, foreground=fg_color, borderwidth=1)
        style.configure('TSpinbox', fieldbackground=entry_bg, foreground=fg_color, borderwidth=1)
        style.configure('TButton', background=button_bg, foreground=fg_color, borderwidth=1)
        style.configure('TPanedWindow', background=bg_color)
        
        # Configure combobox dropdown
        style.map('TCombobox', 
                 selectbackground=[('readonly', select_bg)],
                 selectforeground=[('readonly', fg_color)])
        
        # Configure button hover effects
        style.map('TButton',
                 background=[('active', select_bg)],
                 foreground=[('active', fg_color)])

    # UI construction
    def _build_file_frame(self):
        frm = ttk.LabelFrame(self, text="Data file")
        frm.pack(fill=tk.X, padx=8, pady=8)

        self.csv_path_var = tk.StringVar()
        ttk.Label(frm, text="CSV:").grid(row=0, column=0, padx=6, pady=6, sticky=tk.W)
        self.csv_entry = ttk.Entry(frm, textvariable=self.csv_path_var, width=80)
        self.csv_entry.grid(row=0, column=1, padx=6, pady=6, sticky=tk.W)
        ttk.Button(frm, text="Browse...", command=self.browse_csv).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(frm, text="Load", command=self.load_csv).grid(row=0, column=3, padx=6, pady=6)

        for i in range(4):
            frm.grid_columnconfigure(i, weight=0)

    def _build_options_frame(self):
        frm = ttk.LabelFrame(self, text="Forecast options")
        frm.pack(fill=tk.X, padx=8, pady=8)

        # Level
        ttk.Label(frm, text="Level:").grid(row=0, column=0, padx=6, pady=6, sticky=tk.W)
        self.level_var = tk.StringVar(value="all")
        self.level_cb = ttk.Combobox(frm, textvariable=self.level_var, state="readonly",
                                     values=["all", "province", "municipality", "barangay"]) 
        self.level_cb.grid(row=0, column=1, padx=6, pady=6, sticky=tk.W)
        self.level_cb.bind("<<ComboboxSelected>>", lambda e: self.on_level_change())

        # Province/Municipality/Barangay
        ttk.Label(frm, text="Province:").grid(row=0, column=2, padx=6, pady=6, sticky=tk.W)
        self.prov_var = tk.StringVar()
        self.prov_cb = ttk.Combobox(frm, textvariable=self.prov_var, state="disabled")
        self.prov_cb.grid(row=0, column=3, padx=6, pady=6, sticky=tk.W)
        self.prov_cb.bind("<<ComboboxSelected>>", lambda e: self.on_province_change())

        ttk.Label(frm, text="Municipality:").grid(row=1, column=0, padx=6, pady=6, sticky=tk.W)
        self.mun_var = tk.StringVar()
        self.mun_cb = ttk.Combobox(frm, textvariable=self.mun_var, state="disabled")
        self.mun_cb.grid(row=1, column=1, padx=6, pady=6, sticky=tk.W)
        self.mun_cb.bind("<<ComboboxSelected>>", lambda e: self.on_municipality_change())

        ttk.Label(frm, text="Barangay:").grid(row=1, column=2, padx=6, pady=6, sticky=tk.W)
        self.bgy_var = tk.StringVar()
        self.bgy_cb = ttk.Combobox(frm, textvariable=self.bgy_var, state="disabled")
        self.bgy_cb.grid(row=1, column=3, padx=6, pady=6, sticky=tk.W)

        # Periods
        ttk.Label(frm, text="Forecast months:").grid(row=2, column=0, padx=6, pady=6, sticky=tk.W)
        self.periods_var = tk.IntVar(value=6)
        self.periods_spin = ttk.Spinbox(frm, from_=1, to=60, textvariable=self.periods_var, width=6)
        self.periods_spin.grid(row=2, column=1, padx=6, pady=6, sticky=tk.W)

        # Model
        ttk.Label(frm, text="Model:").grid(row=2, column=2, padx=6, pady=6, sticky=tk.W)
        self.model_var = tk.StringVar(value="both" if PROPHET_AVAILABLE else "sarima")
        model_values = ["both", "sarima", "prophet"] if PROPHET_AVAILABLE else ["sarima"]
        self.model_cb = ttk.Combobox(frm, textvariable=self.model_var, state="readonly", values=model_values)
        self.model_cb.grid(row=2, column=3, padx=6, pady=6, sticky=tk.W)
        if not PROPHET_AVAILABLE:
            self.model_cb.configure(state="readonly")

        # Fill missing
        ttk.Label(frm, text="Fill missing:").grid(row=3, column=0, padx=6, pady=6, sticky=tk.W)
        self.fill_var = tk.StringVar(value="zero")
        self.fill_cb = ttk.Combobox(frm, textvariable=self.fill_var, state="readonly", values=["zero", "ffill"])
        self.fill_cb.grid(row=3, column=1, padx=6, pady=6, sticky=tk.W)

        # Output dir
        ttk.Label(frm, text="Output folder:").grid(row=3, column=2, padx=6, pady=6, sticky=tk.W)
        self.out_dir_var = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "outputs"))
        self.out_dir_entry = ttk.Entry(frm, textvariable=self.out_dir_var, width=40)
        self.out_dir_entry.grid(row=3, column=3, padx=6, pady=6, sticky=tk.W)
        ttk.Button(frm, text="Browse...", command=self.browse_output_dir).grid(row=3, column=4, padx=6, pady=6)

        for i in range(5):
            frm.grid_columnconfigure(i, weight=0)

    def _build_buttons_frame(self):
        frm = ttk.Frame(self)
        frm.pack(fill=tk.X, padx=8, pady=4)

        self.run_btn = ttk.Button(frm, text="Run Forecast", command=self.run_forecast)
        self.run_btn.pack(side=tk.LEFT, padx=4)

        self.open_out_btn = ttk.Button(frm, text="Open Outputs Folder", command=self.open_outputs_folder)
        self.open_out_btn.pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = ttk.Label(frm, textvariable=self.status_var)
        self.status_lbl.pack(side=tk.RIGHT, padx=4)

    def _build_plot_and_preview(self):
        container = ttk.PanedWindow(self, orient=tk.VERTICAL)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Plot frame
        plot_frame = ttk.LabelFrame(container, text="Forecast plot")
        container.add(plot_frame, weight=3)

        self.fig, self.ax = plt.subplots(figsize=(9.5, 3.8), dpi=100, constrained_layout=True)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title("Forecast", color='white')
        self.ax.set_xlabel("Date", color='white')
        self.ax.set_ylabel("Cases", color='white')
        self.ax.tick_params(colors='white')
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax.grid(True, axis="both", alpha=0.3, color='gray')
        self.ax.set_axisbelow(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Preview frame
        preview_frame = ttk.LabelFrame(container, text="Forecast preview (head)")
        container.add(preview_frame, weight=2)

        self.preview_txt = tk.Text(preview_frame, height=12, 
                                  bg="#3c3c3c", fg="#ffffff", 
                                  insertbackground="#ffffff", 
                                  selectbackground="#404040")
        self.preview_txt.pack(fill=tk.BOTH, expand=True)

    # Event handlers and helpers
    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.out_dir_var.set(path)

    def load_csv(self):
        path = self.csv_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return
        try:
            df = read_csv_safely(path, self.date_col)
            # Map actual column names for dropdown population
            self.prov_col = find_col_case_insensitive(df, self.prov_col)
            self.mun_col = find_col_case_insensitive(df, self.mun_col)
            self.bgy_col = find_col_case_insensitive(df, self.bgy_col)
            self.date_col = find_col_case_insensitive(df, self.date_col)
            self.male_col = find_col_case_insensitive(df, self.male_col)
            self.female_col = find_col_case_insensitive(df, self.female_col)

            self.df = df
            self.populate_geos()
            self.status_var.set(f"Loaded {os.path.basename(path)} ({len(df)} rows)")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error loading CSV", str(e))

    def populate_geos(self):
        if self.df is None:
            return
        df = self.df
        provs = sorted(pd.Series(df[self.prov_col].dropna().unique(), dtype=str)) if self.prov_col in df.columns else []
        self.prov_cb.configure(values=provs)
        if provs:
            self.prov_var.set(provs[0])
        else:
            self.prov_var.set("")
        self.update_municipalities()
        self.update_barangays()
        self.on_level_change()

    def on_level_change(self):
        level = self.level_var.get()
        # Enable/disable combos depending on level
        self.prov_cb.configure(state=("readonly" if level in ("province", "municipality", "barangay") else "disabled"))
        self.mun_cb.configure(state=("readonly" if level in ("municipality", "barangay") else "disabled"))
        self.bgy_cb.configure(state=("readonly" if level == "barangay" else "disabled"))

    def on_province_change(self):
        self.update_municipalities()
        self.update_barangays()

    def on_municipality_change(self):
        self.update_barangays()

    def update_municipalities(self):
        if self.df is None or self.mun_col not in self.df.columns:
            self.mun_cb.configure(values=[])
            self.mun_var.set("")
            return
        df = self.df
        if self.prov_var.get():
            df = df[df[self.prov_col] == self.prov_var.get()]
        muns = sorted(pd.Series(df[self.mun_col].dropna().unique(), dtype=str))
        self.mun_cb.configure(values=muns)
        self.mun_var.set(muns[0] if muns else "")

    def update_barangays(self):
        if self.df is None or self.bgy_col not in self.df.columns:
            self.bgy_cb.configure(values=[])
            self.bgy_var.set("")
            return
        df = self.df
        if self.prov_var.get():
            df = df[df[self.prov_col] == self.prov_var.get()]
        if self.mun_var.get():
            df = df[df[self.mun_col] == self.mun_var.get()]
        bgys = sorted(pd.Series(df[self.bgy_col].dropna().unique(), dtype=str))
        self.bgy_cb.configure(values=bgys)
        self.bgy_var.set(bgys[0] if bgys else "")

    def run_forecast(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return

        level = self.level_var.get()
        province = self.prov_var.get() if level in ("province", "municipality", "barangay") else None
        municipality = self.mun_var.get() if level in ("municipality", "barangay") else None
        barangay = self.bgy_var.get() if level == "barangay" else None
        periods = int(self.periods_var.get())
        model = self.model_var.get()
        fill_missing = self.fill_var.get()
        out_dir = self.out_dir_var.get().strip() or os.path.join(os.path.dirname(__file__), "outputs")
        csv_path = self.csv_path_var.get().strip()

        # Validate Prophet availability
        if model in ("both", "prophet") and not PROPHET_AVAILABLE:
            messagebox.showinfo("Prophet not available", "Prophet is not installed. Using SARIMA only.")
            model = "sarima"

        # Disable run while processing
        self.run_btn.configure(state=tk.DISABLED)
        self.status_var.set("Running forecast... this may take a moment.")

        def task():
            try:
                df = read_csv_safely(csv_path, self.date_col)
                y, meta = prepare_series(
                    df=df,
                    date_col=self.date_col,
                    male_col=self.male_col,
                    female_col=self.female_col,
                    level=level,
                    province=province,
                    municipality=municipality,
                    barangay=barangay,
                    fill_missing=fill_missing,
                )
                seasonal_m, freq = infer_seasonal_period_and_freq(y.index)

                outputs: dict[str, pd.DataFrame] = {}
                if model in ("both", "sarima"):
                    sarima_out = forecast_with_sarima(y, periods, seasonal_m)
                    outputs["sarima"] = sarima_out
                if model in ("both", "prophet") and PROPHET_AVAILABLE:
                    prophet_out = forecast_with_prophet(y, periods, freq)
                    outputs["prophet"] = prophet_out

                pieces = ["all"]
                if province:
                    pieces.append(province)
                if municipality:
                    pieces.append(municipality)
                if barangay:
                    pieces.append(barangay)
                base_name = "_".join(pieces).replace(" ", "_")

                paths = save_outputs(out_dir, base_name, y, outputs)
                self.last_output_dir = out_dir

                # Deliver to UI thread
                self.after(0, lambda: self.on_forecast_done(y, outputs, paths))
            except Exception as e:  # noqa: BLE001
                self.after(0, lambda: self.on_forecast_error(str(e)))

        threading.Thread(target=task, daemon=True).start()

    def on_forecast_done(self, y: pd.Series, outputs: dict[str, pd.DataFrame], paths: dict[str, str]):
        # Re-enable run button
        self.run_btn.configure(state=tk.NORMAL)
        self.status_var.set("Done. Outputs saved.")

        # Update plot
        self.ax.clear()
        # Determine x-limits across actuals and forecasts
        all_future_ds = pd.DatetimeIndex([])
        for df in outputs.values():
            try:
                all_future_ds = all_future_ds.append(pd.to_datetime(df["ds"]))
            except Exception:
                pass
        x_min = y.index.min()
        x_max = (max(all_future_ds.max(), y.index.max()) if len(all_future_ds) else y.index.max())

        # Plot actuals
        self.ax.plot(y.index, y.values, marker="o", linewidth=1.8, label="Actual", color="#00d4aa")

        # Plot forecasts and confidence bands
        colors = {"sarima": "#4fc3f7", "prophet": "#ffab40"}
        for name, df in outputs.items():
            df = df.sort_values("ds")
            c = colors.get(name, None)
            self.ax.plot(df["ds"], df["yhat"], marker="o", linewidth=1.8, label=f"{name.upper()} forecast", color=c)
            self.ax.fill_between(df["ds"], df["yhat_lower"], df["yhat_upper"], alpha=0.25, color=c)

        # Mark forecast start (first future month)
        forecast_start = (y.index.max() + pd.offsets.MonthBegin(1)).to_pydatetime()
        self.ax.axvline(forecast_start, color="#ffffff", linestyle=":", alpha=0.7, label="Forecast start")

        # Axis formatting and alignment
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_title("Animal Bite Cases Forecast", color='white')
        self.ax.set_xlabel("Date", color='white')
        self.ax.set_ylabel("Cases", color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, axis="both", alpha=0.3, color='gray')
        self.ax.set_axisbelow(True)
        legend = self.ax.legend(ncol=3, loc="upper left")
        legend.get_frame().set_facecolor('#404040')
        for text in legend.get_texts():
            text.set_color('white')
        self.canvas.draw()

        # Update preview text
        self.preview_txt.delete("1.0", tk.END)
        for name, df in outputs.items():
            self.preview_txt.insert(tk.END, f"\n{name.upper()} forecast (head):\n")
            self.preview_txt.insert(tk.END, df.head().to_string(index=False))
            self.preview_txt.insert(tk.END, "\n")
        self.preview_txt.insert(tk.END, "\nSaved files:\n")
        for k, v in paths.items():
            self.preview_txt.insert(tk.END, f" - {k}: {v}\n")

    def on_forecast_error(self, msg: str):
        self.run_btn.configure(state=tk.NORMAL)
        self.status_var.set("Error")
        messagebox.showerror("Forecast error", msg)

    def open_outputs_folder(self):
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        try:
            # Windows: open File Explorer
            os.startfile(out_dir)  # noqa: P204
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Open folder", str(e))


if __name__ == "__main__":
    # Switch to PyQt6-based GUI
    from forecast_qt import main as qt_main
    raise SystemExit(qt_main())
