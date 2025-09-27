#!/usr/bin/env python3
"""
Forecast animal bite cases using SARIMA and Prophet.

- Loads a CSV with columns: PROV_CODE, MUN_CODE, BGY_CODE, DATE, RAB_ANIMBITE_M, RAB_ANIMBITE_F
- Aggregates to a single time series per date (optionally filtered by province/municipality/barangay)
- Resamples to monthly frequency and forecasts next N months
- Trains SARIMA (statsmodels) with a small grid-search over orders
- Trains Prophet (if installed). If Prophet isn't available, continues with SARIMA only
- Saves CSV outputs and a comparison plot

Usage examples:
    python forecast.py --file "Animal Bites Cases.csv" --periods 6
    python forecast.py --file "Animal Bites Cases.csv" --level municipality --province RIZAL --municipality TAYTAY --periods 12
    python forecast.py --file "Animal Bites Cases.csv" --level barangay --province RIZAL --municipality "CITY OF ANTIPOLO" --barangay "Santa Cruz" --model both --periods 9

Requirements:
    pandas, numpy, statsmodels, matplotlib
    prophet (optional; if installed, it will be used)

Notes:
    - If date parsing is ambiguous (e.g., 01/02/2025), the script tries both day-first and month-first and picks the one that parses more rows.
    - Seasonal period is chosen based on inferred frequency (12 for monthly, 7 for daily, etc.).
"""

import argparse
import os
import warnings
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Prophet is optional
PROPHET_AVAILABLE = False
PROPHET_CLASS = None
try:
    from prophet import Prophet as _Prophet

    PROPHET_AVAILABLE = True
    PROPHET_CLASS = _Prophet
except Exception:  # noqa: BLE001
    try:
        from fbprophet import Prophet as _Prophet  # legacy package name

        PROPHET_AVAILABLE = True
        PROPHET_CLASS = _Prophet
    except Exception:  # noqa: BLE001
        PROPHET_AVAILABLE = False
        PROPHET_CLASS = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast animal bite cases using SARIMA and Prophet")
    p.add_argument("--file", default="Animal Bites Cases.csv", help="Path to the input CSV file")

    p.add_argument("--date-col", default="DATE", help="Name of the date column")
    p.add_argument("--male-col", default="RAB_ANIMBITE_M", help="Name of the male count column")
    p.add_argument("--female-col", default="RAB_ANIMBITE_F", help="Name of the female count column")

    p.add_argument("--level", choices=["all", "province", "municipality", "barangay"], default="all",
                   help="Aggregation level for filtering before aggregation")
    p.add_argument("--province", default=None, help="Province filter (exact match)")
    p.add_argument("--municipality", default=None, help="Municipality/City filter (exact match)")
    p.add_argument("--barangay", default=None, help="Barangay filter (exact match)")

    p.add_argument("--periods", type=int, default=6, help="Number of future months to forecast")
    p.add_argument("--model", choices=["both", "sarima", "prophet"], default="both",
                   help="Which model(s) to run")
    p.add_argument("--fill-missing", choices=["zero", "ffill"], default="zero",
                   help="How to fill missing months after resampling")
    p.add_argument("--output-dir", default="outputs", help="Directory to save outputs")

    return p.parse_args()


def read_csv_safely(path: str, date_col: str) -> pd.DataFrame:
    # Try utf-8-sig first to handle BOM; fall back to utf-8
    for enc in ("utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        # last resort
        df = pd.read_csv(path)

    if date_col not in df.columns:
        # Try case-insensitive match
        candidates = {c.lower(): c for c in df.columns}
        if date_col.lower() in candidates:
            date_col_real = candidates[date_col.lower()]
            df.rename(columns={date_col_real: date_col}, inplace=True)

    return df


def smart_parse_datetime(series: pd.Series) -> pd.Series:
    # Try day-first and month-first; pick the one with fewer NaT
    s1 = pd.to_datetime(series, dayfirst=True, errors="coerce")
    s2 = pd.to_datetime(series, dayfirst=False, errors="coerce")
    n1 = s1.isna().sum()
    n2 = s2.isna().sum()
    return s1 if n1 <= n2 else s2


def prepare_series(
    df: pd.DataFrame,
    date_col: str,
    male_col: str,
    female_col: str,
    level: str,
    province: Optional[str],
    municipality: Optional[str],
    barangay: Optional[str],
    fill_missing: str = "zero",
) -> Tuple[pd.Series, Dict[str, str]]:
    # Normalize column names via case-insensitive remap
    cols = {c.lower(): c for c in df.columns}

    def get_col(name: str) -> str:
        return cols.get(name.lower(), name)

    prov_col = get_col("PROV_CODE")
    mun_col = get_col("MUN_CODE")
    bgy_col = get_col("BGY_CODE")
    date_col = get_col(date_col)
    male_col = get_col(male_col)
    female_col = get_col(female_col)

    # Filters
    if level == "province" and province is None:
        raise ValueError("--province is required when level is province")
    if level in {"municipality", "barangay"} and municipality is None:
        raise ValueError("--municipality is required when level is municipality/barangay")
    if level == "barangay" and barangay is None:
        raise ValueError("--barangay is required when level is barangay")

    df = df.copy()
    # Parse date
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = smart_parse_datetime(df[date_col])
    df = df.dropna(subset=[date_col])

    # Cast numeric
    for c in (male_col, female_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Apply filters
    meta = {}
    if province is not None and prov_col in df.columns:
        df = df[df[prov_col] == province]
        meta["province"] = province
    if municipality is not None and mun_col in df.columns:
        df = df[df[mun_col] == municipality]
        meta["municipality"] = municipality
    if barangay is not None and bgy_col in df.columns:
        df = df[df[bgy_col] == barangay]
        meta["barangay"] = barangay

    # Build total target
    df["TOTAL"] = df[male_col].fillna(0) + df[female_col].fillna(0)

    # Aggregate by date
    s = (
        df.groupby(df[date_col].dt.to_period("M"))["TOTAL"].sum().astype(float)
    )
    # Convert PeriodIndex to TimestampIndex at month start
    s.index = s.index.to_timestamp(how="start")
    s = s.sort_index()

    # Ensure monthly frequency with resample
    s = s.asfreq("MS")

    if s.isna().any():
        if fill_missing == "ffill":
            s = s.ffill().fillna(0)
        else:
            s = s.fillna(0)

    if s.size < 3:
        raise ValueError("Not enough observations after filtering/aggregation. Need at least 3 monthly points.")

    return s, meta


def infer_seasonal_period_and_freq(idx: pd.DatetimeIndex) -> Tuple[int, str]:
    # Monthly data
    return 12, "MS"


def sarima_grid_search(
    y: pd.Series,
    seasonal_m: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    # Modest grid to keep runtime reasonable
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    best_aic = np.inf
    best_order = None
    best_seasonal = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, seasonal_m)
                            try:
                                model = SARIMAX(
                                    y,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                res = model.fit(disp=False)
                                aic = res.aic
                                if np.isfinite(aic) and aic < best_aic:
                                    best_aic = aic
                                    best_order = order
                                    best_seasonal = seasonal_order
                            except Exception:  # noqa: BLE001
                                continue
    if best_order is None:
        # Fallback
        best_order = (1, 1, 1)
        best_seasonal = (0, 1, 1, seasonal_m)

    return best_order, best_seasonal


def forecast_with_sarima(y: pd.Series, periods: int, seasonal_m: int) -> pd.DataFrame:
    order, seasonal_order = sarima_grid_search(y, seasonal_m)
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=periods)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.2)  # 80% interval for comparability with Prophet default

    out = pd.DataFrame(
        {
            "ds": mean.index,
            "yhat": mean.values,
            "yhat_lower": conf.iloc[:, 0].values,
            "yhat_upper": conf.iloc[:, 1].values,
        }
    )
    out["model"] = "sarima"
    return out


def forecast_with_prophet(y: pd.Series, periods: int, freq: str) -> pd.DataFrame:
    if not PROPHET_AVAILABLE or PROPHET_CLASS is None:
        raise RuntimeError("Prophet is not installed. Install 'prophet' to enable Prophet forecasts.")

    dfp = pd.DataFrame({"ds": y.index, "y": y.values})
    m = PROPHET_CLASS(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    m.fit(dfp)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)

    tail = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tail["model"] = "prophet"
    return tail


def save_outputs(
    output_dir: str,
    base_name: str,
    y: pd.Series,
    outputs: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    # Save individual model outputs
    for name, df in outputs.items():
        path = os.path.join(output_dir, f"{base_name}_{name}_forecast.csv")
        df.to_csv(path, index=False)
        paths[name] = path

    # Save combined
    combined = pd.concat(outputs.values(), ignore_index=True)
    combined_path = os.path.join(output_dir, f"{base_name}_combined_forecast.csv")
    combined.sort_values(["ds", "model"]).to_csv(combined_path, index=False)
    paths["combined"] = combined_path

    # Plot
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
        import matplotlib.dates as mdates
        
        # Build a Figure using non-interactive Agg backend (safe in worker threads)
        fig = Figure(figsize=(11, 6), dpi=150, tight_layout=True)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        colors = {"actual": "#00d4aa", "sarima": "#4fc3f7", "prophet": "#ffab40"}
        _canvas = FigureCanvasAgg(fig)

        # Bar settings (compact)
        width_actual_days = 8
        width_forecast_days = 3
        offset_days = {"sarima": -3, "prophet": 3}

        # Plot only forecasts (hide historical actuals)
        x_all = []

        # Forecast bars for each model with error bars
        for name, df in outputs.items():
            df = df.sort_values("ds")
            ds = pd.to_datetime(df["ds"]).dt.to_pydatetime()
            x_f = mdates.date2num(ds) + offset_days.get(name, 0)
            yhat = df["yhat"].to_numpy()
            lower = df["yhat_lower"].to_numpy()
            upper = df["yhat_upper"].to_numpy()
            yerr = [yhat - lower, upper - yhat]
            c = colors.get(name, None)
            ax.bar(x_f, yhat, width=width_forecast_days, color=c, alpha=0.9, label=f"{name.upper()} forecast", align="center")
            ax.errorbar(x_f, yhat, yerr=yerr, fmt="none", ecolor=c, elinewidth=1.2, capsize=3, alpha=0.9)
            if len(x_f):
                x_all.extend(list(x_f))

        # Compute numeric x-limits based on forecast bars only
        if x_all:
            min_num = min(x_all) - width_forecast_days / 2.0
            max_num = max(x_all) + width_forecast_days / 2.0
        else:
            min_num = mdates.date2num(y.index.max())
            max_num = mdates.date2num(y.index.max())

        # Focus view on 2026+ if any forecast months are in/after 2026
        any_2026_plus = any((pd.to_datetime(df["ds"]).dt.year >= 2026).any() for df in outputs.values())
        if any_2026_plus:
            from datetime import datetime as _dt, timedelta as _td
            start_2026_num = mdates.date2num(_dt(2026, 1, 1) - _td(days=15))
            min_num = max(min_num, start_2026_num)

        locator = mdates.AutoDateLocator(minticks=6, maxticks=18)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlim(min_num, max_num)

        ax.set_title("Propsarim Forecast", color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Cases", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, axis="both", alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        ax.margins(x=0.01)
        legend = ax.legend(ncol=3, loc="upper left")
        legend.get_frame().set_facecolor('#404040')
        for text in legend.get_texts():
            text.set_color('white')

        plot_path = os.path.join(output_dir, f"{base_name}_forecast_plot.png")
        fig.savefig(plot_path, dpi=150)
        paths["plot"] = plot_path
    except Exception:  # noqa: BLE001
        pass

    return paths


def main():
    args = parse_args()

    df = read_csv_safely(args.file, args.date_col)

    y, meta = prepare_series(
        df=df,
        date_col=args.date_col,
        male_col=args.male_col,
        female_col=args.female_col,
        level=args.level,
        province=args.province,
        municipality=args.municipality,
        barangay=args.barangay,
        fill_missing=args.fill_missing,
    )

    seasonal_m, freq = infer_seasonal_period_and_freq(y.index)

    # Build a base name for outputs using filters
    pieces = ["all"]
    if meta.get("province"):
        pieces.append(meta["province"])
    if meta.get("municipality"):
        pieces.append(meta["municipality"])
    if meta.get("barangay"):
        pieces.append(meta["barangay"])
    base_name = "_".join(pieces).replace(" ", "_")

    print(f"Loaded {len(y)} monthly points from {y.index.min().date()} to {y.index.max().date()}.")
    print(f"Forecasting {args.periods} future months with seasonal period m={seasonal_m}.")

    outputs: Dict[str, pd.DataFrame] = {}

    if args.model in ("both", "sarima"):
        print("Fitting SARIMA (grid-searching a small set of orders)...")
        sarima_out = forecast_with_sarima(y, args.periods, seasonal_m)
        outputs["sarima"] = sarima_out
        print("SARIMA complete.")

    if args.model in ("both", "prophet"):
        if PROPHET_AVAILABLE and PROPHET_CLASS is not None:
            print("Fitting Prophet...")
            prophet_out = forecast_with_prophet(y, args.periods, freq)
            outputs["prophet"] = prophet_out
            print("Prophet complete.")
        else:
            print(
                "Prophet is not installed. Install 'prophet' to enable Prophet forecasts (continuing without it)."
            )

    if not outputs:
        raise SystemExit("No model outputs produced. Check --model and Prophet installation.")

    paths = save_outputs(args.output_dir, base_name, y, outputs)

    print("\nSaved files:")
    for k, v in paths.items():
        print(f" - {k}: {v}")

    # Show preview
    any_df = next(iter(outputs.values()))
    print("\nForecast preview:")
    print(any_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
