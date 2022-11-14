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

wavelength = np.arange(5, 501, 5)
i_array = []
data_array = []

for i in wavelength:

    print(f"Working with wavelength {i}...")

    # Get fourier and spatial domain
    fourier, spatial = f2s.f2s(i, 0)

    # Calculate local difference
    loc_diff = local_difference.local_activity(spatial, kernel_size = 25)

    i_array.append(i)
    data_array.append(loc_diff)

df_i = pd.DataFrame(i_array)
df_data = pd.DataFrame(data_array)
df = pd.concat([df_i, df_data], join = 'outer', axis = 1)
df.to_csv('df.csv')

print(f"Done!")