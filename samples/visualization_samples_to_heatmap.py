from matrixlib.core import Matrix
from matrixlib.visualization import Visualize


# Create a matrix instance (non-square)
measurements = Matrix([
    [25, 60, 45],  # 25°C, 60% humidity, 45% light
    [24, 65, 50]   # 24°C, 65% humidity, 50% light
])

# Call the improved visualization method
Visualize.to_heatmap(
    measurements,
    title="Environmental Measurements",
    cmap="viridis",
    xlabel="Measurement Type",
    ylabel="Time Period",
    figsize=(10, 6)
)


