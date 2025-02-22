from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from matrixlib.core import Matrix

matplotlib.use('qt5agg')  # Use 'qt5agg' for better Qt support


class Visualize:
    @staticmethod
    def to_heatmap(matrix: Matrix, title="Matrix Heatmap", figsize=None,
                   cmap="coolwarm", show_annotations=True, xlabel=None,
                   ylabel=None, fmt=".2f"):
        """
        Visualize the matrix as a heatmap using seaborn.

        Parameters:
        -----------
        matrix : Matrix
            The matrix object to visualize
        title : str, optional
            Title for the heatmap
        figsize : tuple, optional
            Figure size (width, height) in inches, defaults to (rows, cols)
        cmap : str, optional
            Colormap for the heatmap
        show_annotations : bool, optional
            Whether to display data values on cells
        xlabel : str, optional
            Label for x-axis
        ylabel : str, optional
            Label for y-axis
        fmt : str, optional
            Format string for annotations
        """
        # Check if matrix contains numeric data
        try:
            # Quick test to see if we can perform a numeric operation
            _ = matrix.data[0][0] + 0
        except (TypeError, ValueError):
            raise ValueError("Matrix must contain numeric data for heatmap visualization")

        # Calculate default figure size based on matrix dimensions
        rows, cols = len(matrix.data), len(matrix.data[0])
        if figsize is None:
            figsize = (max(6, cols), max(5, rows))

        plt.figure(figsize=figsize)

        # Create heatmap with optional annotations
        sns.heatmap(matrix.data,
                    annot=show_annotations,
                    fmt=fmt,
                    cmap=cmap,
                    linewidths=0.5)

        # Set title and labels
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def to_graph(matrix: Matrix, directed=True, threshold=0.0, node_labels=None,
                 figsize=(10, 8), title="Graph Representation", layout='spring',
                 node_color='skyblue', edge_color='gray', node_size=500,
                 font_size=10, with_labels=True, edge_width_factor=1.0):
        """
        Visualize the matrix as a graph (network) using networkx without numpy dependency.

        Parameters:
        -----------
        matrix : Matrix
            The matrix object to visualize as a graph (adjacency matrix)
        directed : bool, optional
            If True, create a directed graph, otherwise create an undirected graph
        threshold : float, optional
            Only edges with weights above this threshold will be displayed
        node_labels : list, optional
            Custom labels for nodes. If None, uses 0, 1, 2, ... as labels
        figsize : tuple, optional
            Figure size (width, height) in inches
        title : str, optional
            Title for the graph visualization
        layout : str, optional
            Graph layout algorithm ('spring', 'circular', 'random', 'kamada_kawai', 'shell')
        node_color : str, optional
            Color for nodes
        edge_color : str, optional
            Color for edges
        node_size : int, optional
            Size of nodes
        font_size : int, optional
            Size of node labels
        with_labels : bool, optional
            Whether to display node labels
        edge_width_factor : float, optional
            Factor to multiply edge weights for visual thickness
        """
        # Check if matrix is square (requirement for graph adjacency matrix)
        if len(matrix.data) != len(matrix.data[0]):
            raise ValueError("Graph visualization requires a square matrix (adjacency matrix)")

        # Create the appropriate graph type
        if directed:
            g = nx.DiGraph()
        else:
            g = nx.Graph()

        # Add nodes
        num_nodes = len(matrix.data)
        g.add_nodes_from(range(num_nodes))

        # Add edges with weights above threshold
        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = matrix.data[i][j]
                if weight > threshold:
                    g.add_edge(i, j, weight=weight)

        # Create plot
        plt.figure(figsize=figsize)

        # Select layout algorithm
        if layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'random':
            pos = nx.random_layout(g)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        else:
            pos = nx.spring_layout(g)  # default to spring layout

        # Calculate edge widths based on weights
        edge_widths = []
        for u, v in g.edges():
            edge_widths.append(g[u][v]['weight'] * edge_width_factor)

        # Custom node labels if provided
        if node_labels and len(node_labels) >= num_nodes:
            labels = {i: node_labels[i] for i in range(num_nodes)}
        else:
            labels = {i: str(i) for i in range(num_nodes)}

        # Drawing the graph
        nx.draw_networkx_nodes(g, pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(g, pos, width=edge_widths, edge_color=edge_color,
                               arrowsize=10 if directed else 0)

        if with_labels:
            nx.draw_networkx_labels(g, pos, labels=labels, font_size=font_size)

        #  edge labels (weights)
        edge_labels = {(u, v): f'{g[u][v]["weight"]:.2f}' for u, v in g.edges()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=font_size - 2)

        plt.title(title)
        plt.axis('off')  # Hide axes
        plt.tight_layout()
        plt.show()

        return g

    @staticmethod
    def to_barchart(matrix: Matrix, title="Matrix Bar Chart", figsize=(10, 6), xlabel="Columns", ylabel="Values"):
        """
        Visualize each row of the matrix as a bar chart.

        Parameters:
        -----------
        matrix : Matrix
            The matrix object to visualize.
        title : str, optional
            Title for the bar chart.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        xlabel : str, optional
            Label for x-axis.
        ylabel : str, optional
            Label for y-axis.
        """

        rows, cols = len(matrix.data), len(matrix.data[0])
        x = np.arange(cols)

        plt.figure(figsize=figsize)

        for i in range(rows):
            plt.bar(x, matrix.data[i], alpha=0.6, label=f"Row {i}")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x, labels=[str(i) for i in range(cols)])  # Ensure column labels
        plt.legend()
        plt.show()

    @staticmethod
    def to_surface3d(matrix: Matrix, title="3D Surface Plot", figsize=(10, 8),
                     cmap='viridis', view_angle=(30, 45)):
        """
        Visualize the matrix as a 3D surface plot.

        Parameters:
        -----------
        matrix : Matrix
            The matrix object to visualize
        title : str, optional
            Title for the surface plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        cmap : str, optional
            Colormap for the surface
        view_angle : tuple, optional
            Initial viewing angle (elevation, azimuth) in degrees
        """
        # Create the figure and 3D axes
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Create coordinate grids
        rows, cols = len(matrix.data), len(matrix.data[0])
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))

        # Convert matrix data to proper format
        z = np.array(matrix.data)

        # Create the surface plot
        surf = ax.plot_surface(x, y, z, cmap=cmap,
                               linewidth=0.5, antialiased=True)

        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Set labels and title
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        ax.set_zlabel('Values')
        ax.set_title(title)

        # Set the viewing angle
        ax.view_init(view_angle[0], view_angle[1])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def to_correlation_network(matrix: Matrix, threshold: float = 0.5,
                               figsize: Tuple[int, int] = (10, 10)):
        """
        Visualize matrix as a correlation network where edges represent
        correlations above the threshold.

        Parameters:
        -----------
        matrix : Matrix
            Correlation matrix
        threshold : float
            Minimum correlation value to show edge
        figsize : tuple
            Figure size for the network
        """
        # Create figure and axes with space for colorbar
        fig, ax = plt.subplots(figsize=figsize)

        G = nx.Graph()

        # Add nodes
        for i in range(len(matrix.data)):
            G.add_node(i)

        # Add edges for correlations above threshold
        for i in range(len(matrix.data)):
            for j in range(i + 1, len(matrix.data)):
                if abs(matrix.data[i][j]) >= threshold:
                    G.add_edge(i, j, weight=abs(matrix.data[i][j]))

        pos = nx.spring_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_color='lightblue',
                               node_size=500,
                               ax=ax)

        # Edge colors based on correlation strength
        edges = G.edges()
        if edges:  # Only proceed if there are edges
            weights = [G[u][v]['weight'] for u, v in edges]

            # Create a normalization for the colormap
            norm = plt.Normalize(min(weights), max(weights))

            # Draw edges with colors
            edge_collection = nx.draw_networkx_edges(
                G, pos,
                edge_color=weights,
                edge_cmap=plt.cm.RdYlBu,
                edge_vmin=min(weights),
                edge_vmax=max(weights),
                width=2,
                ax=ax
            )

            # Add labels
            nx.draw_networkx_labels(G, pos, ax=ax)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Correlation Strength")

        ax.set_title("Correlation Network")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
