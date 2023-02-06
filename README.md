# Membrane-fluctuation-analysis
Analysis of the membrane fluctuations from the data of the molecular dynamics (MD) simulation.

My scientific project devoted to the calculation of elastic parameters of the lipid membranes.
The goal of the project was to explain and find the mechanism of the deviation of the theoretical predictions and results obtained from the MD simulations.
To do that I wrote a programm for the analysis of the fluctuations of lipid molecules directions (so-called directors) and compared results with 
perfomed Monte Carlo simulations.

Disclaimer: the work is in progress, thefore the project is not in a final user-friendly form, more in a working form, some comments are made in Russian.

### Files:

- Fluct_analysis_production.py - main programm that relies on the descritization of the space, deviding it on the square cells
- MC_2D_softening_run.py - module for running the Monte Carlo simulations
- fluctuation_analysis_2D.py - analysis of the directors fluctuations by means of Fourier transform
- fluctuation_analysis_2D_continuous.py - analysis of the directors fluctuations by means of continuous Fourier transform
- MC_2D_continuous_production.py - main programm that performs continuous Fourier transform NOT relying on the descritization of the space
- MC_2D_production.py - main program for the analysis of the Monte Carlo simulation results. 
- CG_derectors_to_cells.py - module for transfroming molecular dynamics simulations to the values of the directors.
