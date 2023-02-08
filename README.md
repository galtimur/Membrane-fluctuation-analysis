# Membrane-fluctuation-analysis
Analysis of the membrane fluctuations from the data of the molecular dynamics (MD) simulation.

My scientific project devoted to the calculation of elastic parameters of the lipid membranes.
The goal of the project was to explain and find the mechanism of the deviation of the theoretical predictions and results obtained from the MD simulations.
To do that I wrote a programme for the analysis of the fluctuations of lipid molecules directions (so-called directors) and compared results with 
performed Monte Carlo simulations.

Disclaimer: the work is in progress, therefore the project is not in a final user-friendly form, more in a working form, some comments are made in Russian.

### Files:

- Fluct_analysis_production.py - a main programme that relies on the discretization of the space, dividing it into the square cells
- MC_2D_softening_run.py - module for running the Monte Carlo simulations
- fluctuation_analysis_2D.py - analysis of the directors' fluctuations through Fourier transform
- fluctuation_analysis_2D_continuous.py - analysis of the directors' fluctuations through continuous Fourier transform
- MC_2D_continuous_production.py - main programme that performs continuous Fourier transform NOT relying on the discretization of the space
- MC_2D_production.py - main program for the analysis of the Monte Carlo simulation results. 
- CG_derectors_to_cells.py - module for transforming molecular dynamics simulations to the values of the directors.
