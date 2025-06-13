import imageio
import os
from glob import glob

# Set the total duration (in seconds) for the entire GIF
#total_duration = 3.0
total_duration = 5.0

# Get all .png files in the current directory, sorted alphabetically
png_files = sorted(glob("*.png"))

# Compute duration per frame
num_frames = len(png_files)
if num_frames == 0:
    raise ValueError("No .png files found.")
duration_per_frame = total_duration / num_frames

# Read images
images = [imageio.imread(png) for png in png_files]

# Save as GIF
output_filename = "output.gif"
imageio.mimsave(output_filename, images, duration=duration_per_frame)

print(f"Saved GIF '{output_filename}' with {num_frames} frames over {total_duration} seconds.")

