# Matrix Visualization Class (Visualize)
## Overview

The `Visualize` class provides a comprehensive suite of visualization methods for matrix data. It leverages multiple visualization libraries including matplotlib, seaborn, networkx, and plotly to create various types of visual representations of matrix data.

## Visualization Methods

### 1. Heatmap Visualization (`to_heatmap`)

```python
Visualize.to_heatmap(matrix, title="Matrix Heatmap", figsize=None, cmap="coolwarm", 
                     show_annotations=True, xlabel=None, ylabel=None, fmt=".2f")
```

Creates a color-coded heatmap representation of the matrix data.

**Parameters:**
- `matrix`: Input Matrix object
- `title`: Chart title (default: "Matrix Heatmap")
- `figsize`: Figure size in inches (default: auto-calculated)
- `cmap`: Color map (default: "coolwarm")
- `show_annotations`: Show values in cells (default: True)
- `xlabel/ylabel`: Axis labels
- `fmt`: Number format for annotations (default: ".2f")

**Features:**
- Automatic size scaling based on matrix dimensions
- Customizable color scheme
- Optional value annotations
- Customizable axis labels

**Example Usage:**
```python
matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Visualize.to_heatmap(matrix, title="Temperature Data", cmap="YlOrRd")
```

### 2. Graph Visualization (`to_graph`)

```python
Visualize.to_graph(matrix, directed=True, threshold=0.0, node_labels=None, 
                   figsize=(10, 8), title="Graph Representation", layout='spring')
```

Converts matrix to a network graph representation.

**Parameters:**
- `matrix`: Input adjacency matrix
- `directed`: Create directed/undirected graph (default: True)
- `threshold`: Minimum edge weight to display (default: 0.0)
- `node_labels`: Custom node labels
- `layout`: Graph layout algorithm
- `edge_width_factor`: Edge thickness multiplier

**Layout Options:**
- 'spring': Force-directed layout
- 'circular': Circular layout
- 'random': Random layout
- 'kamada_kawai': Kamada-Kawai algorithm
- 'shell': Shell layout

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

### 3. Bar Chart Visualization (`to_barchart`)

```python
Visualize.to_barchart(matrix, title="Matrix Bar Chart", figsize=(10, 6), 
                      xlabel="Columns", ylabel="Values")
```

Creates a bar chart where each row is represented as a group of bars.

**Parameters:**
- `matrix`: Input Matrix object
- `title`: Chart title
- `figsize`: Figure size in inches
- `xlabel/ylabel`: Axis labels

**Features:**
- Multiple rows shown as grouped bars
- Automatic legend generation
- Customizable colors and transparency
- Adjustable figure size

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

### 4. 3D Surface Plot (`to_surface3d`)

```python
Visualize.to_surface3d(matrix, title="3D Surface Plot", figsize=(10, 8),
                       cmap='viridis', view_angle=(30, 45))
```

Creates a three-dimensional surface representation of the matrix.

**Parameters:**
- `matrix`: Input Matrix object
- `title`: Plot title
- `figsize`: Figure size
- `cmap`: Color map
- `view_angle`: Initial viewing angle (elevation, azimuth)

**Features:**
- Interactive 3D rotation
- Customizable color mapping
- Adjustable viewing angle
- Optional color bar

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

### 5. Correlation Network (`to_correlation_network`)

```python
Visualize.to_correlation_network(matrix, threshold=0.5, figsize=(10, 10))
```

Visualizes correlation matrices as networks.

**Parameters:**
- `matrix`: Correlation matrix
- `threshold`: Minimum correlation to show
- `figsize`: Figure size

**Features:**
- Edge colors represent correlation strength
- Customizable correlation threshold
- Automatic node positioning
- Interactive color bar

**Example Usage:**
```python
corr_matrix = Matrix([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.6],
    [0.3, 0.6, 1.0]
])
Visualize.to_correlation_network(corr_matrix, threshold=0.5)
```

### 6. Sankey Diagram (`to_sankey`)

```python
Visualize.to_sankey(matrix, source_labels=None, target_labels=None,
                    title="Sankey Diagram", figsize=(12, 8))
```

Creates a Sankey diagram showing flows between rows and columns.

**Parameters:**
- `matrix`: Flow matrix
- `source_labels`: Labels for sources
- `target_labels`: Labels for targets
- `source_colors/target_colors`: Custom colors

**Features:**
- Interactive flow visualization
- Customizable node colors
- Automatic label generation
- Responsive sizing

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

## Best Practices

1. **Color Selection:**
   - Use colorblind-friendly palettes
   - Ensure sufficient contrast
   - Choose appropriate color maps for data type

2. **Layout Optimization:**
   - Adjust figure size for data dimensions
   - Consider aspect ratio for readability
   - Use appropriate spacing and margins

3. **Performance Considerations:**
   - Limit annotations for large matrices
   - Use thresholds for dense networks
   - Consider memory usage for 3D plots

4. **Error Handling:**
   - Validate matrix dimensions
   - Check for numeric data
   - Handle edge cases (empty matrices, NaN values)

## References

1. Matplotlib Documentation: https://matplotlib.org/
2. NetworkX Documentation: https://networkx.org/
3. Plotly Documentation: https://plotly.com/python/
4. Seaborn Documentation: https://seaborn.pydata.org/

## Dependencies
- matplotlib
- networkx
- numpy
- seaborn
- plotly