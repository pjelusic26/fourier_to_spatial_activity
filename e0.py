'''
Step 1
    - generate black freq images with dots of various radii
    - result: sin in spatial domain
Step 2
    - find entropy/ local diff in spatial domain
Step 3
    - embed (with fixed IF) and detect WM for each image
Conclusion:
    - find a correlation between image activity (of chosen metric) and WM detection
'''

import numpy as np
import pandas as pd
import f2s
import local_difference
from wmark import WaterMark
import time

# Checkpoint
start = time.time()

# Setup class
wmark = WaterMark(5)

# Define kernel size
mask_size = 5

# Define wavelengths
wavelength = np.arange(3, 1005, 5)

# Create empty lists for data
wavelength_array = []
activity_array = []
detection_array = []
correlation_array = []

for i in wavelength:

    print(f"Working with wavelength {i}...")

    # Get fourier and spatial domain
    fourier, spatial = f2s.f2s(i, 0)

    # Embed mark
    marked, freq_mag = wmark.embedMark(spatial, factor = 250)
    print(f"Mark embedded...")

    # Get correlation values
    decoder = wmark.decodeMark(marked, 'CORR')
    print(f"Decoding done...")

    # Is the watermark detected?
    decision = wmark.detectOutlier(marked, 'CORR', alpha = 0.0001)
    print(f"Watermark is detected: {decision}, with a {round(decoder, 2)} correlation.")

    # Calculate local activity
    diff = local_difference.local_activity(spatial, kernel_size = mask_size)
    print(f"Activity is: {round(diff, 5)}")

    # Add data to lists
    wavelength_array.append(i)
    activity_array.append(diff)
    detection_array.append(decision)
    correlation_array.append(decoder)

# Transform lists into DataFrames
df_wavelength = pd.DataFrame(wavelength_array)
df_activity = pd.DataFrame(activity_array)
df_detection = pd.DataFrame(detection_array)
df_correlation = pd.DataFrame(correlation_array)

# Concatenate DataFrames and name columns
df = pd.concat([df_wavelength, df_activity, df_detection, df_correlation], join = 'outer', axis = 1)
df.columns =['Wavelength', 'Activity', 'Detection', 'Correlation']

# Name and save file
filename = f"df_kernel_{mask_size}.csv"
df.to_csv(filename)

end = time.time()
print(f"Done with script in {round((end-start), 3)} seconds...")