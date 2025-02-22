Here's an extended and improved version of the documentation for your `Visualize` class. I've added more details and explanations for each section and method, and refined the formatting for clarity.

---

# Matrix Visualization Class (Visualize)

## Overview

The `Visualize` class is designed to offer a wide range of visualization methods for matrix data. By utilizing popular visualization libraries such as **matplotlib**, **seaborn**, **networkx**, and **plotly**, this class allows users to explore and represent matrix data in various meaningful and interactive ways. Each visualization type is customizable, allowing for specific use cases and preferences to be met, whether it’s showing correlations, plotting graphs, or visualizing 3D surfaces.

## Visualization Methods

### 1. Heatmap Visualization (`to_heatmap`)

```python
Visualize.to_heatmap(matrix, title="Matrix Heatmap", figsize=None, cmap="coolwarm", 
                     show_annotations=True, xlabel=None, ylabel=None, fmt=".2f")
```

Creates a color-coded heatmap representation of matrix data, making it easy to visualize data patterns and correlations.

**Parameters:**
- `matrix`: The matrix object containing the data to be visualized.
- `title`: Title of the chart (default: "Matrix Heatmap").
- `figsize`: Size of the figure in inches (default: automatically calculated).
- `cmap`: Color map to be used (default: "coolwarm").
- `show_annotations`: Whether to display cell values as annotations (default: True).
- `xlabel`: Label for the x-axis.
- `ylabel`: Label for the y-axis.
- `fmt`: Format string for annotations (default: ".2f").

**Features:**
- Automatic size scaling based on the dimensions of the matrix.
- Customizable color scheme for better readability.
- Optional annotations within cells to display numeric values.
- Customizable axis labels for clarity.

**Example Usage:**
```python
matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Visualize.to_heatmap(matrix, title="Temperature Data", cmap="YlOrRd")
```

---

### 2. Graph Visualization (`to_graph`)

```python
Visualize.to_graph(matrix, directed=True, threshold=0.0, node_labels=None, 
                   figsize=(10, 8), title="Graph Representation", layout='spring')
```

This method converts an adjacency matrix into a network graph, offering a visual representation of the relationships between nodes.

**Parameters:**
- `matrix`: Input adjacency matrix that defines connections between nodes.
- `directed`: Boolean indicating whether the graph should be directed (default: True).
- `threshold`: Minimum edge weight for visualization (default: 0.0).
- `node_labels`: Custom labels for nodes in the graph.
- `layout`: Specifies the layout algorithm for node positioning (e.g., 'spring', 'circular').
- `edge_width_factor`: A multiplier for edge thickness, proportional to edge weight.

**Layout Options:**
- `'spring'`: Force-directed layout (nodes push away from each other).
- `'circular'`: Nodes arranged in a circle.
- `'random'`: Nodes positioned randomly.
- `'kamada_kawai'`: Kamada-Kawai algorithm for more symmetric node placement.
- `'shell'`: Nodes placed in concentric circles.

**Example Usage:**
```python
adjacency_matrix = Matrix([
    [0, 1, 0.5],
    [0.2, 0, 0.8],
    [0.3, 0.4, 0]
])
Visualize.to_graph(adjacency_matrix, 
                   node_labels=['A', 'B', 'C'],
                   layout='circular')
```

<img src="to-graph-test.png" alt="Graph Visualization Example" width="600"/>

---

### 3. Bar Chart Visualization (`to_barchart`)

```python
Visualize.to_barchart(matrix, title="Matrix Bar Chart", figsize=(10, 6), 
                      xlabel="Columns", ylabel="Values")
```

This method visualizes matrix data as a bar chart, with each row of the matrix represented as a group of bars.

**Parameters:**
- `matrix`: Matrix object to be visualized.
- `title`: Title of the chart.
- `figsize`: Size of the figure in inches.
- `xlabel`: Label for the x-axis.
- `ylabel`: Label for the y-axis.

**Features:**
- Multiple rows are grouped into clusters of bars.
- Customizable colors for bars and transparency.
- Automatically generates legends for better clarity.
- Adjustable figure size for better fitting to the data.

**Example Usage:**
```python
data_matrix = Matrix([
    [10, 20, 30, 40],
    [15, 25, 35, 45]
])
Visualize.to_barchart(data_matrix, 
                      title="Monthly Sales Comparison",
                      xlabel="Quarter",
                      ylabel="Revenue")
```

<img src="to_barchart-test.png" alt="Bar Chart Example" width="600"/>

---

### 4. 3D Surface Plot (`to_surface3d`)

```python
Visualize.to_surface3d(matrix, title="3D Surface Plot", figsize=(10, 8),
                       cmap='viridis', view_angle=(30, 45))
```

Creates a 3D surface plot for matrix data, useful for visualizing topographical or volumetric data.

**Parameters:**
- `matrix`: Matrix object to represent in 3D space.
- `title`: Title for the plot.
- `figsize`: Size of the figure.
- `cmap`: Color map to apply to the surface plot.
- `view_angle`: Initial viewing angle (elevation, azimuth).

**Features:**
- Interactive 3D plot for rotating and zooming.
- Customizable color mapping for the surface.
- Adjustable view angle to highlight specific features of the plot.
- Optionally, add a color bar for visual reference.

**Example Usage:**
```python
height_matrix = Matrix([
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [2, 3, 4, 3]
])
Visualize.to_surface3d(height_matrix, 
                       title="Terrain Map",
                       cmap='terrain')
```

<img src="to_3d_test.png" alt="3D Surface Plot Example" width="600"/>

---

### 5. Correlation Network (`to_correlation_network`)

```python
Visualize.to_correlation_network(matrix, threshold=0.5, figsize=(10, 10))
```

Visualizes a correlation matrix as a network graph, where nodes represent variables and edges represent the correlation between them.

**Parameters:**
- `matrix`: Correlation matrix for network visualization.
- `threshold`: Minimum correlation value to display in the network.
- `figsize`: Size of the figure for the network.

**Features:**
- Edge colors represent the strength of the correlation.
- Customizable correlation threshold to limit edges.
- Automatic layout and positioning of nodes.
- Interactive color bar for better understanding of correlation values.

**Example Usage:**
```python
corr_matrix = Matrix([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.6],
    [0.3, 0.6, 1.0]
])
Visualize.to_correlation_network(corr_matrix, threshold=0.5)
```

---

### 6. Sankey Diagram (`to_sankey`)

```python
Visualize.to_sankey(matrix, source_labels=None, target_labels=None,
                    title="Sankey Diagram", figsize=(12, 8))
```

Creates a Sankey diagram to visualize the flow of data between different categories, where the width of the arrows represents the magnitude of flow.

**Parameters:**
- `matrix`: Flow matrix for visualizing the Sankey diagram.
- `source_labels`: Labels for the source categories.
- `target_labels`: Labels for the target categories.
- `source_colors/target_colors`: Optional custom colors for source and target nodes.

**Features:**
- Interactive flow visualization with adjustable node sizes.
- Customizable color schemes for source and target nodes.
- Automatic label generation based on matrix values.
- Resizable chart based on the number of categories.

**Example Usage:**
```python
flow_matrix = Matrix([
    [10, 20, 30],
    [15, 25, 35]
])
Visualize.to_sankey(flow_matrix,
                    source_labels=['Source A', 'Source B'],
                    target_labels=['Target X', 'Target Y', 'Target Z'])
```

<img src="newplot.png" alt="Sankey Diagram Example" width="600"/>

---

## Integration Examples

### Combined Visualization Example

```python
# Create a sample matrix
data = Matrix([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.6],
    [0.3, 0.6, 1.0]
])

# Create multiple visualizations
Visualize.to_heatmap(data, title="Correlation Heatmap")
Visualize.to_graph(data, title="Network Representation")
Visualize.to_correlation_network(data, threshold=0.5)
```

---

### Analysis Pipeline Example

```python
def analyze_matrix(matrix: Matrix):
    """Comprehensive matrix visualization analysis"""
    # 1. Basic structure visualization
    Visualize.to_heatmap(matrix, title="Data Overview")
    
    # 2. Network analysis
    Visualize.to_graph(matrix, threshold=0.3)
    
    # 3. Distribution analysis
    Visualize.to_barchart(matrix)
    
    # 4. Surface analysis
    Visualize.to_surface3d(matrix)
    
    # 5. Flow analysis
    Visualize.to_sankey(matrix)
```

---

## Best Practices

1. **Color Selection:**
   - Always choose colorblind-friendly palettes, such as **Color Universal Design (CUD)**.
   - Ensure there is sufficient contrast between foreground and background.
   - Choose color maps appropriate for the type of data (e.g., **sequential** for ordered data and **diverging** for data with both positive and negative values).

2. **Layout Optimization:**
   - Adjust figure size based on the matrix’s dimension and visualization complexity.
   - Maintain a proper aspect ratio for readability, especially in graphs and heatmaps.
   - Use adequate spacing between elements, including labels and titles.

3. **Performance Considerations:**
   - For large matrices, limit the annotations to avoid performance issues.
   - Apply thresholds to reduce the complexity of dense network visualizations.
   - For 3D surface plots, reduce the matrix size or number of points to optimize rendering.

4. **Error Handling:**
   - Validate matrix dimensions before proceeding with visualization.
   - Ensure all data in the matrix is numeric, as some visualizations may fail with non-numeric data.
   - Handle edge cases like empty matrices or matrices with NaN values.

---

## References

1. [Matplotlib Documentation](https://matplotlib.org/)
2. [NetworkX Documentation](https://networkx.org/)
3. [Plotly Documentation](https://plotly.com/python/)
4. [Seaborn Documentation](https://seaborn.pydata.org/)

---

## Dependencies
- **matplotlib**
- **networkx**
- **numpy**
- **seaborn**
- **plotly**

---

This documentation should provide a clear understanding of the `Visualize` class and its functionalities, along with examples and best practices for its use. Let me know if you need further adjustments or additional sections!
