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

## Guided adjustment tests

The core explanation and residuals appear above the plots. “Test adjustment directions” runs the displayed model plus positive and negative perturbations of luminosity and temperature (default 1%). The table lists solver status, stopping radius, residuals, and changes in absolute residuals. Compare stopping radii before interpreting changes; smaller residuals alone do not establish convergence. Results are hidden when the displayed model or adjustment size changes. Numerical errors are distinguished from core boundary mismatches.

## Compact plots

Four default panels combine normalized temperature/density, fractional mass/luminosity, nuclear energy generation, and energy transport. Actual values are available on hover. Pressure, opacity and d ln P/d ln T are in Advanced plots. Red bands and solid lines indicate invalid computed shells; dashed red lines identify the innermost finite positive-radius shell in failed models. Markers apply to the current model. Invalid mass coordinates may reverse, so that axis uses lines only.

## Grid search and zoom

Open Grid search in luminosity and temperature near the top. Set ranges and sample counts (default 25×25, maximum 200×200), optionally use logarithmic luminosity spacing, and run the grid. The status map and downloadable table report solver outcomes and residuals. Select a trial and click Inspect selected grid trial to display its profiles. With a displayed model, the grid holds its mass and composition fixed. Otherwise it uses sidebar mass and composition. Narrow the bounds to refine a promising region.

Limit x-range sets minimum and maximum coordinates for all main and advanced plots; logarithmic bounds must be positive. The artificial surface shell (index 1, maximum radius, zero temperature/pressure/density) is exempt from invalid-shell markers. Interior invalid shells remain marked.

## Numba acceleration

The numerical integration and grid loop compile with Numba. The grid computes compact diagnostics instead of constructing profile tables for every cell; full profiles are built when a trial is inspected. No fastmath, adaptive stepping or physics changes were introduced. The first call after restart has compilation overhead; subsequent calls use compiled code and repeated grids are cached.

Local warm benchmark for 225 models: 4.60 s with the Python reference versus 0.064 s with the compiled grid (about 72×). Deployment hardware and display overhead affect total user-visible time. All flags, shell counts and profile columns matched the reference across 67 test models.

`starmodel.py` remains the reference. `fast_kernel.py` is generated from its numerical functions by `python scripts/build_fast_solver.py`; regenerate and run `python -m unittest discover -s tests` after solver changes. `fast_solver.py` reconstructs profile output; `fast_grid.py` handles compiled grid summaries.

## Smaller step transition

The intermediate inward step is now R/200 rather than R/100: the initial R/1000 step increases by a factor of five when enclosed mass drops below 99%. The core step rule is unchanged. Both Python and Numba versions use this rule. Results can shift slightly: the initial notebook trial no longer passes the density check; a nearby passing trial is L=0.8598766667 L☉, Teff=5496.866667 K at M=1 M☉, X=0.70, Z=0.008. Earlier benchmark timings predate this step change.

## Current integration spacing

The main step stays at R/1000 through the surface and interior; it never increases at the 99%-mass threshold. Near the center, the existing half-remaining-radius rule reduces it to R/5000. The shell limit is 5000. Both reference and compiled solvers use these settings. Earlier step-transition descriptions and timings above document prior versions. Previously accepted input parameters may need retuning with the smaller steps.
