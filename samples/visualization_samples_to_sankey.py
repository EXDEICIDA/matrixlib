from matrixlib.core import Matrix
from matrixlib.visualization import Visualize

# Create a matrix representing flow of resources between departments
resource_flow = Matrix([
    [100, 50, 30],    # Department A's resource distribution
    [20, 150, 80],    # Department B's resource distribution
    [60, 40, 120]     # Department C's resource distribution
])

# Define custom colors
source_colors = ["#1f77b4", "#2ca02c", "#9467bd"]  # Blue, Green, Purple
target_colors = ["#ff7f0e", "#d62728", "#8c564b"]  # Orange, Red, Brown

# Visualize the resource flow with custom colors
Visualize.to_sankey(
    resource_flow,
    source_labels=["Dept A", "Dept B", "Dept C"],
    target_labels=["Project X", "Project Y", "Project Z"],
    title="Resource Flow Between Departments and Projects",
    figsize=(12, 8),
    source_colors=source_colors,
    target_colors=target_colors
)