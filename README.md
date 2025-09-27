# Propsarim: A Hybrid Weekly Forecaster Combining SARIMA and Prophet

Propsarim is a hybrid forecasting toolkit for incident counts (e.g., animal bite cases). It aggregates your data to weekly totals, fits SARIMA and Prophet on a shared training horizon, and combines their forecasts via an inverse-RMSE weighted ensemble. The GUI shows the weekly data you’ll model (“Show Data”) and then plots only future forecasts for SARIMA, Prophet, and the Hybrid. A monthly CLI script is also included for batch workflows.

- GUI: forecast_gui.py (PyQt6)
- Weekly hybrid pipeline: implemented inside the GUI’s worker
- CLI (monthly): forecast.py (saves plots/CSVs)
- Launcher: run_forecast_gui.bat


## Quick Start

1) Install dependencies

- Recommended: create a virtual environment
  - Windows (PowerShell):
    - python -m venv .venv
    - .\.venv\Scripts\Activate.ps1
  - Install deps:
    - pip install -r requirements.txt

Notes on Prophet:
- Installing prophet may take time (CmdStan toolchain setup via cmdstanpy). On Windows, ensure Microsoft C++ Build Tools are installed.

2) Launch the GUI
- Double-click run_forecast_gui.bat
  or
- python D:\prophet_sarima\forecast_gui.py

3) Use the app
- Show Data: Select level (all/municipality/barangay) and filters, then click “Show Data” to display weekly aggregation (W-MON) on the chart and in the preview.
- Run Forecast: Set Forecast weeks (h). Click “Run Forecast” to train on the past (excluding last h weeks), compare SARIMA and Prophet on the last h weeks, combine them (Hybrid) by inverse-RMSE weights, and display only the future bars.

4) CLI (optional, monthly)
- Example overall monthly run (6 months):
  - python D:\prophet_sarima\forecast.py --file "D:\prophet_sarima\Animal Bites Cases.csv" --periods 6
- Municipality monthly run (12 months):
  - python D:\prophet_sarima\forecast.py --file "D:\prophet_sarima\Animal Bites Cases.csv" --level municipality --province RIZAL --municipality TAYTAY --periods 12


## Pipeline Overview (Graph)

```mermaid
flowchart LR
  A[Raw Records (DATE, M/F counts, geography)] --> B[Parse DATE, numeric cast]
  B --> C[Filter (municipality/barangay)]
  C --> D[Weekly aggregate (W-MON), fill missing weeks]
  D --> E[Train/Test Split (last h weeks = test)]
  E --> F1[SARIMA (1,1,1)(1,1,1,52)]
  E --> F2[Prophet (weekly + yearly seasonality)]
  F1 --> G1[Forecast h weeks]
  F2 --> G2[Forecast h weeks]
  G1 --> H[Evaluate on test]
  G2 --> H
  H --> I[Inverse-RMSE weighted Hybrid]
  I --> J[Future-only plot: SARIMA | Prophet | Hybrid]
```


## Mathematical Formulation

### Notation
- Let y_t denote weekly totals at week t (regular W-MON index).
- Backshift operator B: By_t = y_{t-1}. Seasonal period s = 52 (weekly, approx.).

### SARIMA
A general seasonal ARIMA can be written as

$$
\Phi(B)\,\Phi_s(B^s)\,(1 - B)^d\,(1 - B^s)^D\,y_t
\;=\; \Theta(B)\,\Theta_s(B^s)\,\varepsilon_t
\quad \text{with } \varepsilon_t\sim \mathcal{WN}(0,\sigma^2),
$$

where
- Non-seasonal AR polynomial: \(\Phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p\)
- Seasonal AR polynomial: \(\Phi_s(B^s) = 1 - \Phi_1 B^s - \dots - \Phi_P B^{Ps}\)
- Non-seasonal MA polynomial: \(\Theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q\)
- Seasonal MA polynomial: \(\Theta_s(B^s) = 1 + \Theta_1 B^s + \dots + \Theta_Q B^{Qs}\)

In Propsarim, a practical starting specification is

$$
(p,d,q) = (1,1,1),\qquad (P,D,Q,s) = (1,1,1,52),
$$

which handles linear dynamics plus a weekly seasonal component.

### Prophet
Prophet models

$$
 y(t) = g(t) + s(t) + h(t) + \varepsilon_t,
$$

where g(t) is a piecewise linear trend with changepoints, s(t) is a sum of seasonalities (Fourier series), and h(t) optional holidays. For a seasonality of period P with order N,

$$
 s(t) = \sum_{n=1}^{N} \left[ a_n\cos\!\left(\tfrac{2\pi n t}{P}\right) + b_n\sin\!\left(\tfrac{2\pi n t}{P}\right) \right].
$$

Piecewise linear trend with changepoints \(\{\tau_k\}\) (indicator vector \(\mathbf{a}(t)\)) can be written as

$$
 g(t) = \left(k + \mathbf{a}(t)^\top \boldsymbol{\delta}\right)\,t + \left(m + \mathbf{a}(t)^\top \boldsymbol{\gamma}\right),
$$

with priors encouraging sparse changepoints.

We enable weekly and yearly seasonality and fit Prophet on the same training horizon as SARIMA.

### Hybrid: Inverse-RMSE Weighting
Let \(\widehat{y}^{\,(S)}_t\) denote SARIMA’s forecast and \(\widehat{y}^{\,(P)}_t\) denote Prophet’s forecast, both aligned on the h-step validation horizon \(\{t_1,\dots,t_h\}\). Define validation RMSEs

$$
 \operatorname{RMSE}_S = \sqrt{\tfrac{1}{h}\sum_{i=1}^{h}\bigl(y_{t_i} - \widehat{y}^{\,(S)}_{t_i}\bigr)^2},
 \qquad
 \operatorname{RMSE}_P = \sqrt{\tfrac{1}{h}\sum_{i=1}^{h}\bigl(y_{t_i} - \widehat{y}^{\,(P)}_{t_i}\bigr)^2}.
$$

The inverse-RMSE weights are

$$
 w_S = \frac{\tfrac{1}{\operatorname{RMSE}_S}}{\tfrac{1}{\operatorname{RMSE}_S} + \tfrac{1}{\operatorname{RMSE}_P}},
 \qquad
 w_P = 1 - w_S,
$$

and the hybrid forecast is

$$
 \widehat{y}^{\,(H)}_t = w_S\,\widehat{y}^{\,(S)}_t + w_P\,\widehat{y}^{\,(P)}_t.
$$

Confidence bands for Hybrid can be approximated by a weighted combination of component intervals:

$$
 [\widehat{y}^{\,(H)}_t]_{\text{lower}} \approx w_S\,[\widehat{y}^{\,(S)}_t]_{\text{lower}} + w_P\,[\widehat{y}^{\,(P)}_t]_{\text{lower}},
 \quad
 [\widehat{y}^{\,(H)}_t]_{\text{upper}} \approx w_S\,[\widehat{y}^{\,(S)}_t]_{\text{upper}} + w_P\,[\widehat{y}^{\,(P)}_t]_{\text{upper}}.
$$

This assumes weak dependence between model errors; it serves as a pragmatic approximation in practice.


## Evaluation and Final Forecast
- Split: last h weeks serve as the validation horizon; training uses all prior weeks.
- Metrics: RMSE, MAE are reported for SARIMA, Prophet (if installed), and Hybrid.
- Final forecast: After validating, you may retrain on the full weekly series and forecast the next h weeks; the GUI’s “Run Forecast” horizon directly shows future-only bars for the next h weeks.


## Example Outputs (Graphs)

- GUI “Run Forecast”: Future-only bar chart, side-by-side bars for SARIMA (blue), Prophet (orange), Hybrid (lavender), with error bars. Use a large enough “Forecast weeks” to reach 2026+.
- CLI monthly plot: Running the CLI saves a PNG inside outputs/, e.g. outputs/all_forecast_plot.png.

You can embed your latest saved figure in downstream docs by referencing it, for example:

```markdown
![Propsarim Hybrid Forecast](outputs/all_forecast_plot.png)
```


## Advanced Notes
- Variance Stabilization: You may apply \(\log(1+y)\) to training and invert via \(\exp(\cdot)-1\) at forecast time to stabilize variance (not enabled by default).
- Model Selection: The SARIMA order used here is a strong baseline for weekly data. For production, you can grid-search or use information criteria (AICc) on a rolling window.
- Seasonality: Weekly (s≈52) is natural for incident reporting; Prophet’s yearly seasonality can complement calendar effects.
- Uncertainty: Hybrid intervals are heuristic; a more principled approach would model joint uncertainty, but is beyond scope.


## File Inventory
- forecast_gui.py – PyQt6 GUI for weekly hybrid pipeline (Show Data + Run Forecast)
- forecast_qt.py – GUI module with plotting/utilities
- forecast.py – Monthly CLI script (saves CSV + PNG)
- run_forecast_gui.bat – Windows launcher for the GUI
- requirements.txt – Dependencies (numpy, pandas, scipy, statsmodels, matplotlib, pyqt6, prophet, cmdstanpy)
- outputs/ – Created at runtime for plots and CSVs


## License
This repository is provided for internal and personal use. You may adapt and extend Propsarim to your data and operational needs.
