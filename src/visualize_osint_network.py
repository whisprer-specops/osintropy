"""
OSINTropy Network Visualizer
Creates 3 types of network visualizations:
1. Static NetworkX + Matplotlib (2D publication-quality)
2. Interactive Pyvis (HTML with physics simulation)
3. Interactive Plotly (3D rotating graph)
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Check for optional dependencies
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("⚠️  Pyvis not installed. Run: pip install pyvis")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not installed. Run: pip install plotly")


def load_network_data(filepath='network_graph.json'):
    """Load network graph from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_networkx_graph(network_data):
    """Convert JSON network data to NetworkX graph."""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_id, node_data in network_data['nodes'].items():
        G.add_node(
            node_id,
            node_type=node_data['type'],
            data=node_data['data'],
            sources=node_data.get('sources', [])
        )
    
    # Add edges
    for edge in network_data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            relationship=edge['type'],
            weight=edge['weight']
        )
    
    return G


def visualize_matplotlib(network_data, output_file='network_matplotlib.png'):
    """
    Create static 2D visualization using NetworkX + Matplotlib.
    Publication-quality output with custom styling.
    """
    print("\n🎨 Creating NetworkX + Matplotlib visualization...")
    
    G = create_networkx_graph(network_data)
    
    # Create figure with nice styling
    plt.figure(figsize=(16, 12), facecolor='white')
    plt.title('OSINTropy Network Graph', fontsize=20, fontweight='bold', pad=20)
    
    # Use spring layout for nice spacing
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Define colors for different node types
    node_colors = {
        'person': '#FF6B6B',      # Red
        'phone': '#4ECDC4',       # Teal
        'location': '#95E1D3',    # Light teal
        'organization': '#FFE66D', # Yellow
        'email': '#C7CEEA'        # Light purple
    }
    
    # Define sizes for different node types
    node_sizes = {
        'person': 3000,
        'phone': 2000,
        'location': 2500,
        'organization': 2800,
        'email': 1800
    }
    
    # Draw nodes by type
    for node_type in node_colors.keys():
        nodes_of_type = [n for n, d in G.nodes(data=True) 
                        if d['node_type'] == node_type]
        if nodes_of_type:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes_of_type,
                node_color=node_colors[node_type],
                node_size=node_sizes.get(node_type, 2000),
                alpha=0.9,
                edgecolors='black',
                linewidths=2
            )
    
    # Draw edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        width=weights,
        alpha=0.5,
        edge_color='gray'
    )
    
    # Draw labels
    labels = {}
    for node_id, node_data in network_data['nodes'].items():
        data = node_data['data']
        if node_data['type'] == 'person':
            labels[node_id] = data.get('name', 'Unknown')
        elif node_data['type'] == 'phone':
            labels[node_id] = data.get('number', 'Unknown')
        elif node_data['type'] == 'location':
            addr = data.get('address', 'Unknown')
            # Truncate long addresses
            labels[node_id] = addr[:25] + '...' if len(addr) > 25 else addr
        elif node_data['type'] == 'organization':
            labels[node_id] = data.get('name', 'Unknown')
        else:
            labels[node_id] = str(node_id)[:8]
    
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=10,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # Add edge labels for relationship types
    edge_labels = {(u, v): G[u][v]['relationship'] 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels,
        font_size=8,
        font_color='darkblue',
        alpha=0.7
    )
    
    # Add legend
    legend_elements = []
    for node_type, color in node_colors.items():
        count = sum(1 for n, d in G.nodes(data=True) 
                   if d['node_type'] == node_type)
        if count > 0:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, markersize=15,
                          label=f'{node_type.capitalize()} ({count})',
                          markeredgecolor='black', markeredgewidth=2)
            )
    
    plt.legend(handles=legend_elements, loc='upper left',
              fontsize=12, framealpha=0.9)
    
    # Add stats box
    stats_text = (
        f"Nodes: {G.number_of_nodes()}\n"
        f"Edges: {G.number_of_edges()}\n"
        f"Density: {nx.density(G):.3f}"
    )
    plt.text(0.02, 0.02, stats_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {output_file}")
    plt.close()


def visualize_pyvis(network_data, output_file='network_interactive.html'):
    """
    Create interactive HTML visualization using Pyvis.
    Features physics simulation and interactive exploration.
    """
    if not PYVIS_AVAILABLE:
        print("❌ Pyvis not available. Skipping...")
        return
    
    print("\n🌐 Creating Pyvis interactive visualization...")
    
    # Create network with custom settings
    net = Network(
        height='800px',
        width='100%',
        bgcolor='#ffffff',
        font_color='black',
        heading='OSINTropy Interactive Network Graph'
    )
    
    # Enable physics for nice simulation
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.001,
        damping=0.09
    )
    
    # Color mapping
    color_map = {
        'person': '#FF6B6B',
        'phone': '#4ECDC4',
        'location': '#95E1D3',
        'organization': '#FFE66D',
        'email': '#C7CEEA'
    }
    
    # Size mapping
    size_map = {
        'person': 25,
        'phone': 20,
        'location': 22,
        'organization': 24,
        'email': 18
    }
    
    # Add nodes
    for node_id, node_data in network_data['nodes'].items():
        node_type = node_data['type']
        data = node_data['data']
        
        # Create label
        if node_type == 'person':
            label = data.get('name', 'Unknown')
            title = f"<b>{label}</b><br>"
            if 'age' in data:
                title += f"Age: {data['age']}<br>"
            if 'phones' in data:
                title += f"Phones: {len(data['phones'])}<br>"
            if 'relatives' in data:
                title += f"Relatives: {', '.join(data['relatives'])}<br>"
        elif node_type == 'phone':
            label = data.get('number', 'Unknown')
            title = f"<b>Phone:</b> {label}"
        elif node_type == 'location':
            label = data.get('address', 'Unknown')[:30]
            title = f"<b>Location:</b><br>{data.get('address', 'Unknown')}"
        elif node_type == 'organization':
            label = data.get('name', 'Unknown')
            title = f"<b>Organization:</b> {label}"
        else:
            label = str(node_id)[:10]
            title = str(node_data)
        
        # Add sources info
        sources = node_data.get('sources', [])
        if sources:
            title += f"<br><b>Sources:</b> {', '.join(sources)}"
        
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color_map.get(node_type, '#CCCCCC'),
            size=size_map.get(node_type, 20),
            shape='dot'
        )
    
    # Add edges
    for edge in network_data['edges']:
        net.add_edge(
            edge['source'],
            edge['target'],
            title=f"{edge['type']} (weight: {edge['weight']:.2f})",
            weight=edge['weight'],
            color='gray'
        )
    
    # Add controls
    net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    
    # Save
    net.save_graph(output_file)
    print(f"✅ Saved to: {output_file}")
    print(f"   Open in browser to interact!")


def visualize_plotly_3d(network_data, output_file='network_3d.html'):
    """
    Create 3D interactive visualization using Plotly.
    Allows rotation and zoom for spatial exploration.
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available. Skipping...")
        return
    
    print("\n🎲 Creating Plotly 3D visualization...")
    
    G = create_networkx_graph(network_data)
    
    # Generate 3D layout using spring layout
    # First get 2D layout, then add z-dimension
    pos_2d = nx.spring_layout(G, dim=2, k=2, iterations=50, seed=42)
    
    # Create 3D positions
    pos_3d = {}
    for node_id, (x, y) in pos_2d.items():
        # Add z-coordinate based on node type
        node_type = G.nodes[node_id]['node_type']
        z_levels = {
            'person': 1.0,
            'phone': 0.5,
            'location': 0.0,
            'organization': 0.75,
            'email': 0.25
        }
        z = z_levels.get(node_type, 0.5)
        pos_3d[node_id] = (x, y, z)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in network_data['edges']:
        x0, y0, z0 = pos_3d[edge['source']]
        x1, y1, z1 = pos_3d[edge['target']]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none',
        showlegend=False
    )
    
    # Prepare node traces by type
    node_traces = []
    color_map = {
        'person': '#FF6B6B',
        'phone': '#4ECDC4',
        'location': '#95E1D3',
        'organization': '#FFE66D',
        'email': '#C7CEEA'
    }
    
    for node_type, color in color_map.items():
        nodes_of_type = [n for n, d in G.nodes(data=True) 
                        if d['node_type'] == node_type]
        
        if not nodes_of_type:
            continue
        
        node_x = [pos_3d[n][0] for n in nodes_of_type]
        node_y = [pos_3d[n][1] for n in nodes_of_type]
        node_z = [pos_3d[n][2] for n in nodes_of_type]
        
        # Create labels
        node_text = []
        for node_id in nodes_of_type:
            data = network_data['nodes'][node_id]['data']
            if node_type == 'person':
                text = f"<b>{data.get('name', 'Unknown')}</b>"
                if 'age' in data:
                    text += f"<br>Age: {data['age']}"
            elif node_type == 'phone':
                text = f"Phone: {data.get('number', 'Unknown')}"
            elif node_type == 'location':
                text = f"Location:<br>{data.get('address', 'Unknown')}"
            elif node_type == 'organization':
                text = f"Org: {data.get('name', 'Unknown')}"
            else:
                text = str(node_id)[:15]
            
            sources = network_data['nodes'][node_id].get('sources', [])
            if sources:
                text += f"<br>Sources: {', '.join(sources)}"
            
            node_text.append(text)
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            name=node_type.capitalize(),
            marker=dict(
                size=12,
                color=color,
                line=dict(color='black', width=1)
            ),
            text=[network_data['nodes'][n]['data'].get('name', 
                  network_data['nodes'][n]['data'].get('number', 
                  str(n)[:8])) for n in nodes_of_type],
            textposition='top center',
            textfont=dict(size=10),
            hovertext=node_text,
            hoverinfo='text'
        )
        node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='OSINTropy 3D Network Graph',
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='black')
        ),
        showlegend=True,
        hovermode='closest',
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='white'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    
    # Save
    fig.write_html(output_file)
    print(f"✅ Saved to: {output_file}")
    print(f"   Open in browser to rotate and explore!")


def main():
    """Main execution - creates all three visualizations."""
    print("=" * 80)
    print("OSINTropy Network Visualizer")
    print("=" * 80)
    
    # Load data
    try:
        network_data = load_network_data('network_graph.json')
        print(f"\n📊 Loaded network with:")
        print(f"   Nodes: {network_data['stats']['node_count']}")
        print(f"   Edges: {network_data['stats']['edge_count']}")
    except FileNotFoundError:
        print("❌ Error: network_graph.json not found!")
        print("   Run example_usage_script.py first to generate network data.")
        return
    except Exception as e:
        print(f"❌ Error loading network data: {e}")
        return
    
    # Create all visualizations
    visualize_matplotlib(network_data)
    visualize_pyvis(network_data)
    visualize_plotly_3d(network_data)
    
    print("\n" + "=" * 80)
    print("✨ Visualization Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • network_matplotlib.png - Static high-quality image")
    print("  • network_interactive.html - Interactive physics simulation (Pyvis)")
    print("  • network_3d.html - 3D rotating graph (Plotly)")
    print("\n💡 Tip: Open .html files in your browser for interactive exploration!")
    print("=" * 80)


if __name__ == "__main__":
    main()
