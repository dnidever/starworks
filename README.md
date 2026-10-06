# StarWorks

**An interactive stellar structure lab.**

A Streamlit teaching app wrapping David Nidever's Python STATSTAR solver.

## Run locally

Use Python 3.10 or newer. In this folder run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The browser opens a local app. Click Run model to calculate. Keep a model as a comparison before running another one. Downloads contain integrated shells in cgs units, plus fractional radius, mass and luminosity.

## Solver provenance and changes

Bundled solver is based on the starmodel.py uploaded August 25, 2026, rather than a verified latest starmodel2.py checkout. The source notebook said those modules were equivalent. Use your latest solver for future scientific validation.

Two changes were made in the bundled copy:

* Core step transition uses abs(deltar) >= 0.5*r rather than deltar >= 0.5*r. Steps are negative for inward integration, so the original comparison could not trigger at positive radius.
* Text output is optional (output_path keyword), avoiding a shared starmodl_py.dat file when multiple users run the app. File handles are closed.

Physics formulas and solver acceptance checks are otherwise retained. An accepted solver flag is not an independent accuracy guarantee. Failed trial models remain viewable and are labeled. Extrapolated core points are excluded from plots, and residual mass/luminosity are reported separately. Model comparisons use each model's own normalized coordinates.

The solver outputs total nuclear energy generation only; this version does not invent separate pp/CNO contributions. It models homogeneous main-sequence stars, not stellar evolution. The initial parameters (1, 0.86071, 5500.2, 0.70, 0.008) come from the notebook.

## Files

* app.py — user interface
* model_runner.py — input validation and solver adapter
* starmodel.py — bundled solver
* requirements.txt — dependencies

This is a local runnable app; it has not been published to Streamlit Cloud. For Streamlit Community Cloud, use repository `dnidever/starworks`, branch `main`, and entry point `app.py`.
