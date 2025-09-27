#!/usr/bin/env python3
"""
Propsarim — PyQt6 GUI for hybrid weekly forecasting (SARIMA + Prophet).

- Choose CSV, level (all/province/municipality/barangay), filters, model, horizon
- Run forecast on a worker thread and view a Matplotlib plot
- Saves outputs (CSVs + plot) using the same logic as forecast.py

Requires: PyQt6, pandas, numpy, statsmodels, matplotlib; prophet (optional)

Run:
    python forecast_gui.py     (batch file already points here by importing this module)
    or
    python forecast_qt.py
"""

import os
import sys
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QFileDialog,
    QTextEdit,
    QSplitter,
    QGroupBox,
    QStatusBar,
    QMessageBox,
)

# Dark matplotlib theme
plt.style.use("dark_background")

# Import forecasting logic from forecast.py
try:
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


def _smart_parse(series: pd.Series) -> pd.Series:
    s1 = pd.to_datetime(series, errors='coerce', dayfirst=True)
    s2 = pd.to_datetime(series, errors='coerce', dayfirst=False)
    return s1 if s1.isna().sum() <= s2.isna().sum() else s2


class ForecastWorker(QObject):
    success = pyqtSignal(object, object, object)  # y: pd.Series, outputs: dict, paths: dict
    error = pyqtSignal(str)

    def __init__(
        self,
        csv_path: str,
        date_col: str,
        mun_col: str,
        bgy_col: str,
        male_col: str,
        female_col: str,
        level: str,
        province: Optional[str],
        municipality: Optional[str],
        barangay: Optional[str],
        periods: int,
        fill_missing: str,
        out_dir: str,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.date_col = date_col
        self.mun_col = mun_col
        self.bgy_col = bgy_col
        self.male_col = male_col
        self.female_col = female_col
        self.level = level
        self.province = province
        self.municipality = municipality
        self.barangay = barangay
        self.periods = periods
        self.fill_missing = fill_missing
        self.out_dir = out_dir

    def run(self) -> None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            Pcls = None
            if PROPHET_AVAILABLE:
                try:
                    from prophet import Prophet as Pcls
                except Exception:
                    try:
                        from fbprophet import Prophet as Pcls
                    except Exception:
                        Pcls = None

            df = read_csv_safely(self.csv_path, self.date_col)
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get(self.date_col.lower(), self.date_col)
            mun_col = cols.get(self.mun_col.lower(), self.mun_col)
            bgy_col = cols.get(self.bgy_col.lower(), self.bgy_col)
            male_col = cols.get(self.male_col.lower(), self.male_col)
            female_col = cols.get(self.female_col.lower(), self.female_col)

            # Parse dates smartly
            s1 = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            s2 = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)
            df[date_col] = s1.where(s1.notna(), s2)
            df = df.dropna(subset=[date_col]).copy()

            df[male_col] = pd.to_numeric(df[male_col], errors='coerce').fillna(0)
            df[female_col] = pd.to_numeric(df[female_col], errors='coerce').fillna(0)

            # Filters
            if self.municipality and mun_col in df.columns:
                df = df[df[mun_col] == self.municipality]
            if self.barangay and bgy_col in df.columns:
                df = df[df[bgy_col] == self.barangay]

            df['TOTAL'] = df[male_col] + df[female_col]
            df = df[[date_col, 'TOTAL']].set_index(date_col).sort_index()

            # Weekly aggregation and fill
            y_w = df['TOTAL'].resample('W-MON').sum()
            y_w = y_w.asfreq('W-MON').fillna(0)

            h = max(1, int(self.periods))
            if len(y_w) <= h + 5:
                raise ValueError('Not enough weekly data to forecast. Increase data or reduce forecast weeks.')

            y_train = y_w.iloc[:-h]
            y_test = y_w.iloc[-h:]

            # SARIMA (weekly seasonality 52)
            sarima_model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,52),
                                   enforce_stationarity=False, enforce_invertibility=False)
            sarima_res = sarima_model.fit(disp=False)
            sarima_fc = sarima_res.get_forecast(steps=h)
            sarima_mean = sarima_fc.predicted_mean
            sarima_conf = sarima_fc.conf_int(alpha=0.2)

            # Prophet
            prophet_mean = None; prophet_lower = None; prophet_upper = None
            if Pcls is not None:
                dfp = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})
                m = Pcls(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, seasonality_mode='additive')
                m.fit(dfp)
                future = m.make_future_dataframe(periods=h, freq='W-MON')
                fc = m.predict(future)
                fc = fc.set_index('ds').loc[y_test.index]
                prophet_mean = fc['yhat']
                prophet_lower = fc['yhat_lower']
                prophet_upper = fc['yhat_upper']

            # Metrics and hybrid
            def _rmse(a, b):
                a = np.asarray(a); b = np.asarray(b)
                return float(np.sqrt(np.mean((a - b) ** 2)))
            def _mae(a, b):
                a = np.asarray(a); b = np.asarray(b)
                return float(np.mean(np.abs(a - b)))

            rmse_sarima = _rmse(y_test.values, sarima_mean.values)
            mae_sarima = _mae(y_test.values, sarima_mean.values)

            outputs: dict[str, pd.DataFrame] = {}
            out_sarima = pd.DataFrame({'ds': y_test.index, 'yhat': sarima_mean.values,
                                       'yhat_lower': sarima_conf.iloc[:,0].values, 'yhat_upper': sarima_conf.iloc[:,1].values})
            outputs['sarima'] = out_sarima

            rmse_prophet = None; mae_prophet = None
            if prophet_mean is not None:
                rmse_prophet = _rmse(y_test.values, prophet_mean.values)
                mae_prophet = _mae(y_test.values, prophet_mean.values)
                out_prophet = pd.DataFrame({'ds': y_test.index, 'yhat': prophet_mean.values,
                                            'yhat_lower': prophet_lower.values, 'yhat_upper': prophet_upper.values})
                outputs['prophet'] = out_prophet

            # Weights
            if prophet_mean is not None and rmse_prophet and rmse_prophet > 0 and rmse_sarima > 0:
                w_s = (1.0 / rmse_sarima) / ((1.0 / rmse_sarima) + (1.0 / rmse_prophet))
            else:
                w_s = 1.0
            w_p = 1.0 - w_s

            hyb = w_s * sarima_mean.values + w_p * (prophet_mean.values if prophet_mean is not None else 0)
            # Hybrid intervals as weighted of components when available
            hyb_lower = w_s * sarima_conf.iloc[:,0].values + (w_p * prophet_lower.values if prophet_lower is not None else 0)
            hyb_upper = w_s * sarima_conf.iloc[:,1].values + (w_p * prophet_upper.values if prophet_upper is not None else 0)
            out_hybrid = pd.DataFrame({'ds': y_test.index, 'yhat': hyb,
                                       'yhat_lower': hyb_lower, 'yhat_upper': hyb_upper})
            outputs['hybrid'] = out_hybrid

            rmse_hybrid = _rmse(y_test.values, hyb)
            mae_hybrid = _mae(y_test.values, hyb)

            info = {
                'rmse_sarima': rmse_sarima, 'mae_sarima': mae_sarima,
                'rmse_prophet': rmse_prophet, 'mae_prophet': mae_prophet,
                'rmse_hybrid': rmse_hybrid, 'mae_hybrid': mae_hybrid,
                'w_sarima': w_s, 'w_prophet': w_p,
            }

            self.success.emit(y_w, outputs, info)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Propsarim — Hybrid Weekly Forecast")
        self.resize(1100, 750)

        # Data holders
        self.df: Optional[pd.DataFrame] = None
        self.prov_col = "PROV_CODE"
        self.mun_col = "MUN_CODE"
        self.bgy_col = "BGY_CODE"
        self.date_col = "DATE"
        self.male_col = "RAB_ANIMBITE_M"
        self.female_col = "RAB_ANIMBITE_F"

        self._init_ui()
        self._apply_dark_palette()
        self.statusBar().showMessage("Ready")

        # Try to auto-load default CSV in same folder
        default_csv = os.path.join(os.path.dirname(__file__), "Animal Bites Cases.csv")
        if os.path.exists(default_csv):
            self.csvLine.setText(default_csv)
            self.load_csv()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # File group
        file_group = QGroupBox("Data file")
        fg = QGridLayout(file_group)
        self.csvLine = QLineEdit()
        browseBtn = QPushButton("Browse…")
        loadBtn = QPushButton("Load")
        fg.addWidget(QLabel("CSV:"), 0, 0)
        fg.addWidget(self.csvLine, 0, 1)
        fg.addWidget(browseBtn, 0, 2)
        fg.addWidget(loadBtn, 0, 3)
        root_layout.addWidget(file_group)

        browseBtn.clicked.connect(self.browse_csv)
        loadBtn.clicked.connect(self.load_csv)

        # Options group
        opt_group = QGroupBox("Forecast options")
        og = QGridLayout(opt_group)

        self.levelCombo = QComboBox(); self.levelCombo.addItems(["all", "municipality", "barangay"])
        self.munCombo = QComboBox(); self.munCombo.setEnabled(False)
        self.bgyCombo = QComboBox(); self.bgyCombo.setEnabled(False)
        self.periodsSpin = QSpinBox(); self.periodsSpin.setRange(1, 60); self.periodsSpin.setValue(6)
        self.fillCombo = QComboBox(); self.fillCombo.addItems(["zero", "ffill"])
        self.outDirLine = QLineEdit(os.path.join(os.path.dirname(__file__), "outputs"))
        outBrowseBtn = QPushButton("Browse…")

        # Layout options
        og.addWidget(QLabel("Level:"), 0, 0); og.addWidget(self.levelCombo, 0, 1)
        og.addWidget(QLabel("Municipality:"), 0, 2); og.addWidget(self.munCombo, 0, 3)
        og.addWidget(QLabel("Barangay:"), 1, 0); og.addWidget(self.bgyCombo, 1, 1)
        og.addWidget(QLabel("Forecast weeks:"), 1, 2); og.addWidget(self.periodsSpin, 1, 3)
        og.addWidget(QLabel("Fill missing:"), 2, 2); og.addWidget(self.fillCombo, 2, 3)
        og.addWidget(QLabel("Output folder:"), 3, 0); og.addWidget(self.outDirLine, 3, 1); og.addWidget(outBrowseBtn, 3, 2)

        root_layout.addWidget(opt_group)

        self.levelCombo.currentTextChanged.connect(self.on_level_change)
        self.munCombo.currentTextChanged.connect(self.on_municipality_change)
        outBrowseBtn.clicked.connect(self.browse_output_dir)

        # Apply initial enabled/disabled state for combos
        self.on_level_change()

        # Buttons row
        btn_row = QHBoxLayout()
        self.runBtn = QPushButton("Run Forecast")
        self.showDataBtn = QPushButton("Show Data")
        self.savePlotBtn = QPushButton("Save Forecast Plot")
        btn_row.addWidget(self.runBtn)
        btn_row.addWidget(self.showDataBtn)
        btn_row.addWidget(self.savePlotBtn)
        root_layout.addLayout(btn_row)

        self.runBtn.clicked.connect(self.run_forecast)
        self.showDataBtn.clicked.connect(self.show_data)
        self.savePlotBtn.clicked.connect(self.save_plot)

        # Splitter with plot and preview
        splitter = QSplitter(Qt.Orientation.Vertical)
        # Plot container
        plot_container = QWidget(); plot_layout = QVBoxLayout(plot_container); plot_layout.setContentsMargins(6,6,6,6)
        self.fig, self.ax = plt.subplots(figsize=(9.5, 3.8), dpi=100, constrained_layout=True)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title("Propsarim", color='white')
        self.ax.set_xlabel("Date", color='white')
        self.ax.set_ylabel("Cases", color='white')
        self.ax.tick_params(colors='white')
        locator = mdates.AutoDateLocator(minticks=6, maxticks=18)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax.grid(True, axis="both", alpha=0.3, color='gray')
        self.ax.set_axisbelow(True)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_container)

        # Preview container
        preview_container = QWidget(); pv_layout = QVBoxLayout(preview_container); pv_layout.setContentsMargins(6,6,6,6)
        self.preview = QTextEdit(); self.preview.setReadOnly(True)
        self.preview.setStyleSheet("QTextEdit { background-color: #3c3c3c; color: white; }")
        pv_layout.addWidget(self.preview)
        splitter.addWidget(preview_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

        # Status bar
        self.setStatusBar(QStatusBar())

    def _apply_dark_palette(self) -> None:
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(60, 60, 60))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(64, 64, 64))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(64, 64, 64))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        self.setPalette(palette)

    # Event handlers
    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", os.getcwd(), "CSV files (*.csv);;All files (*.*)")
        if path:
            self.csvLine.setText(path)

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", os.getcwd())
        if path:
            self.outDirLine.setText(path)

    def load_csv(self) -> None:
        path = self.csvLine.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.critical(self, "Error", "Please select a valid CSV file.")
            return
        try:
            df = read_csv_safely(path, self.date_col)
            self.prov_col = find_col_case_insensitive(df, self.prov_col)
            self.mun_col = find_col_case_insensitive(df, self.mun_col)
            self.bgy_col = find_col_case_insensitive(df, self.bgy_col)
            self.date_col = find_col_case_insensitive(df, self.date_col)
            self.male_col = find_col_case_insensitive(df, self.male_col)
            self.female_col = find_col_case_insensitive(df, self.female_col)
            self.df = df
            self.populate_geos()
            self.statusBar().showMessage(f"Loaded {os.path.basename(path)} ({len(df)} rows)")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error loading CSV", str(e))

    def populate_geos(self) -> None:
        if self.df is None:
            return
        df = self.df
        self.update_municipalities()
        self.update_barangays()
        # Ensure controls reflect current level selection
        self.on_level_change()

    def on_level_change(self, *_args) -> None:
        level = self.levelCombo.currentText()
        munE = level in ("municipality", "barangay")
        bgyE = level == "barangay"
        self.munCombo.setEnabled(munE)
        self.bgyCombo.setEnabled(bgyE)
        # Clear selections for disabled combos to avoid stale filters
        if not munE:
            self.munCombo.setCurrentIndex(-1)
        if not bgyE:
            self.bgyCombo.setCurrentIndex(-1)


    def on_municipality_change(self, *_args) -> None:
        self.update_barangays()

    def update_municipalities(self) -> None:
        if self.df is None or self.mun_col not in self.df.columns:
            self.munCombo.clear()
            return
        df = self.df
        muns = sorted(pd.Series(df[self.mun_col].dropna().unique(), dtype=str))
        self.munCombo.clear(); self.munCombo.addItems(muns)

    def update_barangays(self) -> None:
        if self.df is None or self.bgy_col not in self.df.columns:
            self.bgyCombo.clear()
            return
        df = self.df
        if self.munCombo.currentText():
            df = df[df[self.mun_col] == self.munCombo.currentText()]
        bgys = sorted(pd.Series(df[self.bgy_col].dropna().unique(), dtype=str))
        self.bgyCombo.clear(); self.bgyCombo.addItems(bgys)

    def run_forecast(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a CSV first.")
            return
        level = self.levelCombo.currentText()
        province = None
        municipality = self.munCombo.currentText() if level in ("municipality", "barangay") else None
        barangay = self.bgyCombo.currentText() if level == "barangay" else None
        periods = int(self.periodsSpin.value())
        fill_missing = self.fillCombo.currentText()
        out_dir = self.outDirLine.text().strip() or os.path.join(os.path.dirname(__file__), "outputs")
        csv_path = self.csvLine.text().strip()

        self.runBtn.setEnabled(False)
        self.statusBar().showMessage("Running forecast... this may take a moment.")

        self.thread = QThread(self)
        self.worker = ForecastWorker(
            csv_path, self.date_col, self.mun_col, self.bgy_col, self.male_col, self.female_col,
            level, province, municipality, barangay,
            periods, fill_missing, out_dir,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.success.connect(self.on_forecast_done)
        self.worker.error.connect(self.on_forecast_error)
        self.worker.success.connect(lambda *_: self.thread.quit())
        self.worker.error.connect(lambda *_: self.thread.quit())
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(lambda: self.runBtn.setEnabled(True))
        self.thread.start()

    def on_forecast_done(self, y: pd.Series, outputs: dict, paths: dict) -> None:
        self.statusBar().showMessage("Done. Outputs saved.")

        # Plot only future forecasts (hide historical actuals)
        self.ax.clear()

        # Bar chart settings
        colors = {"sarima": "#4fc3f7", "hybrid": "#b39ddb", "prophet": "#ffab40"}
        width_forecast_days = 3
        offset_days = {"sarima": -3, "hybrid": 0, "prophet": 3}

        # Forecast bars with error bars side-by-side
        x_all = []
        for name, df in outputs.items():
            df = df.sort_values("ds")
            ds = pd.to_datetime(df["ds"]).dt.to_pydatetime()
            x_f = mdates.date2num(ds) + offset_days.get(name, 0)
            yhat = df["yhat"].to_numpy()
            lower = df["yhat_lower"].to_numpy()
            upper = df["yhat_upper"].to_numpy()
            yerr = [yhat - lower, upper - yhat]
            c = colors.get(name, None)
            self.ax.bar(x_f, yhat, width=width_forecast_days, color=c, alpha=0.9, label=f"{name.upper()} forecast", align="center")
            self.ax.errorbar(x_f, yhat, yerr=yerr, fmt="none", ecolor=c, elinewidth=1.2, capsize=3, alpha=0.9)
            if len(x_f):
                x_all.extend(list(x_f))

        # Compute numeric x limits with padding and 2026+ focus
        if x_all:
            min_num = min(x_all) - width_forecast_days / 2.0
            max_num = max(x_all) + width_forecast_days / 2.0
        else:
            # Fallback to today's month if somehow empty
            from datetime import datetime as _dt
            min_num = mdates.date2num(_dt.today())
            max_num = mdates.date2num(_dt.today())

        # If there are forecast months in/after 2026, focus the x-axis on 2026+
        any_2026_plus = any((pd.to_datetime(df["ds"]).dt.year >= 2026).any() for df in outputs.values())
        if any_2026_plus:
            start_2026_num = mdates.date2num(datetime(2026, 1, 1) - timedelta(days=15))
            min_num = max(min_num, start_2026_num)

        # Axis formatting
        locator = mdates.AutoDateLocator(minticks=6, maxticks=18)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax.set_xlim(min_num, max_num)
        self.ax.set_title("Propsarim Forecast", color='white')
        self.ax.set_xlabel("Date", color='white')
        self.ax.set_ylabel("Cases", color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, axis="both", alpha=0.3, color='gray')
        self.ax.set_axisbelow(True)
        self.ax.margins(x=0.01)
        legend = self.ax.legend(ncol=3, loc="upper left")
        legend.get_frame().set_facecolor('#404040')
        for text in legend.get_texts():
            text.set_color('white')
        self.canvas.draw()

        # Preview
        self.preview.clear()
        for name, df in outputs.items():
            self.preview.append(f"\n{name.upper()} forecast (head):\n")
            self.preview.append(df.head().to_string(index=False))
        self.preview.append("\nMetrics:\n")
        order_keys = ['rmse_sarima','mae_sarima','rmse_prophet','mae_prophet','rmse_hybrid','mae_hybrid','w_sarima','w_prophet']
        for k in order_keys:
            if isinstance(paths, dict) and k in paths and paths[k] is not None:
                try:
                    val = float(paths[k])
                    self.preview.append(f" - {k}: {val:.3f}")
                except Exception:
                    self.preview.append(f" - {k}: {paths[k]}")

    def on_forecast_error(self, msg: str) -> None:
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Forecast error", msg)

    def show_data(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a CSV first.")
            return
        # Build the series used for forecasting based on current options
        level = self.levelCombo.currentText()
        province = None
        municipality = self.munCombo.currentText() if level in ("municipality", "barangay") else None
        barangay = self.bgyCombo.currentText() if level == "barangay" else None
        fill_missing = self.fillCombo.currentText()
        csv_path = self.csvLine.text().strip()
        try:
            df = read_csv_safely(csv_path, self.date_col)
            # Case-insensitive column mapping
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get(self.date_col.lower(), self.date_col)
            mun_col = cols.get(self.mun_col.lower(), self.mun_col)
            bgy_col = cols.get(self.bgy_col.lower(), self.bgy_col)
            male_col = cols.get(self.male_col.lower(), self.male_col)
            female_col = cols.get(self.female_col.lower(), self.female_col)

            # Parse dates smartly
            df[date_col] = _smart_parse(df[date_col])
            df = df.dropna(subset=[date_col]).copy()

            # Numeric
            df[male_col] = pd.to_numeric(df[male_col], errors='coerce').fillna(0)
            df[female_col] = pd.to_numeric(df[female_col], errors='coerce').fillna(0)

            # Filters
            if municipality and mun_col in df.columns:
                df = df[df[mun_col] == municipality]
            if barangay and bgy_col in df.columns:
                df = df[df[bgy_col] == barangay]

            df['TOTAL'] = df[male_col] + df[female_col]
            df = df[[date_col, 'TOTAL']]
            df = df.set_index(date_col).sort_index()

            # Weekly aggregation (Monday start), fill missing with 0
            y_w = df['TOTAL'].resample('W-MON').sum()
            y_w = y_w.asfreq('W-MON').fillna(0)

            # Preview
            self.preview.clear()
            self.preview.append('Data (weekly aggregated) used for forecasting:\n')
            self.preview.append(f"Range: {y_w.index.min().date()} to {y_w.index.max().date()}  (n={len(y_w)})\n")
            df_show = pd.DataFrame({'ds': y_w.index, 'y': y_w.values})
            if len(df_show) <= 200:
                self.preview.append(df_show.to_string(index=False))
            else:
                self.preview.append(df_show.head(100).to_string(index=False))
                self.preview.append('\n...\n')
                self.preview.append(df_show.tail(100).to_string(index=False))

            # Plot the weekly data on the graph
            self.ax.clear()
            x = mdates.date2num(y_w.index.to_pydatetime())
            self.ax.bar(x, y_w.values, width=5, color='#00d4aa', alpha=0.9, label='Data (weekly)')
            locator = mdates.AutoDateLocator(minticks=6, maxticks=18)
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            if len(x):
                self.ax.set_xlim(min(x) - 3, max(x) + 3)
            self.ax.set_title('Propsarim Weekly Data', color='white')
            self.ax.set_xlabel('Week', color='white')
            self.ax.set_ylabel('Cases', color='white')
            self.ax.tick_params(colors='white')
            self.ax.grid(True, axis='both', alpha=0.3, color='gray')
            self.ax.set_axisbelow(True)
            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, 'Show Data', str(e))

    def save_plot(self) -> None:
        # Save the current figure to PNG in the outputs folder by default
        default_dir = self.outDirLine.text().strip() or os.path.join(os.path.dirname(__file__), "outputs")
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            pass
        default_path = os.path.join(default_dir, "propsarim_forecast.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save Forecast Plot", default_path, "PNG Image (*.png);;All Files (*.*)")
        if not path:
            return
        root, ext = os.path.splitext(path)
        if not ext:
            path = root + ".png"
        try:
            self.fig.savefig(path, dpi=150, facecolor=self.fig.get_facecolor(), edgecolor="none")
            self.statusBar().showMessage(f"Saved plot: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save Plot", str(e))


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
