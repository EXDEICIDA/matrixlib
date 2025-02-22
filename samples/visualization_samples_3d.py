from matrixlib.core import Matrix
from matrixlib.visualization import Visualize

# Create a larger matrix with 24 hours of environmental measurements
# Each row represents one hour (24 hours total)
# Columns: Temperature (°C), Humidity (%), Light (%), Air Quality (PPM), Noise (dB)
measurements = Matrix([
    [20, 65, 0, 30, 35],  # 12 AM
    [19, 68, 0, 28, 30],  # 1 AM
    [18, 70, 0, 25, 28],  # 2 AM
    [18, 72, 0, 25, 25],  # 3 AM
    [17, 75, 0, 24, 25],  # 4 AM
    [17, 75, 10, 25, 30],  # 5 AM
    [18, 72, 30, 28, 40],  # 6 AM
    [20, 68, 60, 35, 55],  # 7 AM
    [22, 65, 80, 40, 65],  # 8 AM
    [24, 60, 90, 45, 70],  # 9 AM
    [26, 55, 95, 48, 72],  # 10 AM
    [27, 50, 100, 50, 75],  # 11 AM
    [28, 48, 100, 52, 75],  # 12 PM
    [29, 45, 100, 55, 78],  # 1 PM
    [30, 42, 95, 53, 76],  # 2 PM
    [29, 45, 90, 50, 75],  # 3 PM
    [28, 50, 80, 48, 73],  # 4 PM
    [27, 55, 60, 45, 70],  # 5 PM
    [25, 60, 40, 40, 68],  # 6 PM
    [23, 65, 20, 35, 60],  # 7 PM
    [22, 68, 0, 32, 55],  # 8 PM
    [21, 70, 0, 30, 45],  # 9 PM
    [20, 72, 0, 28, 40],  # 10 PM
    [20, 70, 0, 25, 35],  # 11 PM
])

# Create 3D surface plot
Visualize.to_surface3d(
    measurements,
    title="24-Hour Environmental Measurements",
    figsize=(12, 8),
    cmap="viridis",
    view_angle=(30, 45)
)

# Also show as heatmap for comparison
Visualize.to_heatmap(
    measurements,
    title="24-Hour Environmental Measurements Heatmap",
    cmap="viridis",
    xlabel="Measurements (Temp, Humidity, Light, Air Quality, Noise)",
    ylabel="Hour of Day",
    figsize=(12, 8)
)
