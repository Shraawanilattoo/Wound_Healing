import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Given data
time = np.array([1, 5, 7, 10])  # Known time points
wound = np.array([0.0, 23.4533, 31.69214, 39.72313])  # Known wound values

# Create the interpolation function
interp_function = interp1d(time, wound, kind='linear', fill_value="extrapolate")

# Generate the complete range of time (from 1 to 10)
time_full = np.arange(1, 11)

# Interpolate the wound values for the full time range
wound_full = interp_function(time_full)

# Print the interpolated wound values
print(wound_full)