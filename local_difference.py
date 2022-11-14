import numpy as np
import pandas as pd
from PIL import Image
import math

def local_activity(channel, kernel_size):

    # Read CMYK image
    # img = np.array(Image.open(filepath))

    # Extract K channel
    # channel = img[:, :, -1]

    # Define start and end points of scanning
    scan_start = math.floor(kernel_size / 2)
    scan_end = channel.shape[0] - math.floor(kernel_size / 2)
    # print(f"Image will be scanned from {scan_start} to {scan_end}.")

    # Define surrounding pixel area
    kernel_start = math.floor(kernel_size / 2)
    kernel_end = math.ceil(kernel_size / 2)
    # print(f"Kernel starts {kernel_start} pixels before, and ends {kernel_end} pixels after the position.")

    data_array = []
    # print(f"Data array shape: {data_array.shape}")

    # Counter for allocating data in the array
    kernel_counter = 0

    # Iterate through image
    for i in range(scan_start, (scan_end + 1)):

        kernel_counter += 1

        for j in range(scan_start, (scan_end + 1)):
            # Define central pixel position
            x, y = i, j
            # print(f"Current center position: {x, y}")

            kernel_area = channel[x-kernel_start : x+kernel_end, y-kernel_start : y+kernel_end]
            diff = channel[x, y] - np.mean(kernel_area)
            data_array.append(diff)
            # print(f"Local Difference: {diff}")

            kernel_counter += 1

    # print(f"How many kernels? {kernel_counter}")
    data_array = pd.DataFrame(data_array)
    data_array.dropna(inplace = True)
    # print(f"Data Array output: {data_array.shape}")

    return data_array.std()[0]
    
