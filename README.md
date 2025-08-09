Satellite Solar Array Orientation Optimization
This script estimates the optimal solar array orientation for satellites by matching observed apparent magnitudes with BRDF-based brightness simulations using LUMOS and numerical optimization.

How it works
Loads observation data from CSV.

Computes Sun’s position for each observation.

Models satellite surfaces with BRDFs (Phong, Binomial, Lambertian).

Optimizes solar array Euler angles per observation via scipy.optimize.minimize.

Outputs calculated magnitudes, orientation angles, and error metrics.

Generates comparison plots.
