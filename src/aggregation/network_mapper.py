"""
Network mapper for visualizing relationships and connections between entities.
Creates network graphs from OSINT data with entropy-weighted edges.
"""

import json
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import hashlib

from utils.logger import get_logger

logger = get_logger(__name__)


class NetworkMapper:
    """
    Maps relationships between entities (people, locations, organizations)
    discovered through OSINT with entropy-based weighting.
    """
    
    def __init__(self):
        """Initialize network mapper."""
        self.nodes = {}  # {node_id: node_data}
        self.edges = []  # [(source, target, weight, relationship_type)]
        self.entity_index = defaultdict(set)  # {entity_value: {node_ids}}
        
    def add_person(self, person_data: Dict[str, Any], source: str) -> str:
        """
        Add person node to network.
        
        Args:
            person_data: Person information dictionary
            source: Data source name
            
        Returns:
            Node ID
        """
        # Generate unique node ID
        node_id = self._generate_node_id(person_data, 'person')
        
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                'type': 'person',
                'data': person_data,
                'sources': [source],
                'entropy_score': person_data.get('entropy_score', 0.0)
            }
        else:
            # Update existing node with new data
            if source not in self.nodes[node_id]['sources']:
                self.nodes[node_id]['sources'].append(source)
            self._merge_data(self.nodes[node_id]['data'], person_data)
        
        # Index key entities
        self._index_entity(node_id, person_data)
        
        return node_id
    
    def add_location(self, location: str, associated_entities: List[str]) -> str:
        """
        Add location node to network.
        
        Args:
            location: Location string
            associated_entities: List of entity IDs associated with location
            
        Returns:
            Location node ID
        """
        node_id = self._generate_node_id({'location': location}, 'location')
        
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                'type': 'location',
                'data': {'address': location},
                'sources': [],
                'associated_count': 0
            }
        
        self.nodes[node_id]['associated_count'] = len(associated_entities)
        
        # Create edges to associated entities
        for entity_id in associated_entities:
            self.add_edge(entity_id, node_id, 'located_at', weight=0.7)
        
        return node_id
    
    def add_organization(self, org_name: str, org_data: Dict[str, Any]) -> str:
        """
        Add organization node.
        
        Args:
            org_name: Organization name
            org_data: Organization data
            
        Returns:
            Node ID
        """
        node_id = self._generate_node_id({'name': org_name}, 'organization')
        
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                'type': 'organization',
                'data': org_data,
                'sources': []
            }
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, 
                 relationship_type: str, weight: float = 1.0,
                 metadata: Optional[Dict] = None):
        """
        Add relationship edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship (relative, associate, etc.)
            weight: Edge weight (0.0-1.0)
            metadata: Optional edge metadata
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge: node not found ({source_id} -> {target_id})")
            return
        
        edge = {
            'source': source_id,
            'target': target_id,
            'type': relationship_type,
            'weight': weight,
            'metadata': metadata or {}
        }
        
        # Check for duplicate
        edge_exists = any(
            e['source'] == source_id and 
            e['target'] == target_id and 
            e['type'] == relationship_type
            for e in self.edges
        )
        
        if not edge_exists:
            self.edges.append(edge)
    
    def map_relationships(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map all relationships from aggregated OSINT data.
        
        Args:
            aggregated_data: Aggregated data from OSINTAggregator
            
        Returns:
            Network graph dictionary
        """
        logger.info("Mapping entity relationships...")
        
        # Process each data source
        for source, source_data in aggregated_data.get('sources', {}).items():
            for record in source_data.get('records', []):
                # Add primary entity
                primary_id = self.add_person(record, source)
                
                # Map relatives
                relatives = record.get('relatives', []) or record.get('associates', [])
                for relative in relatives:
                    if isinstance(relative, str):
                        rel_id = self.add_person({'name': relative}, source)
                        self.add_edge(primary_id, rel_id, 'relative', weight=0.8)
                
                # Map locations
                addresses = record.get('addresses', []) or record.get('locations', [])
                if isinstance(addresses, str):
                    addresses = [addresses]
                    
                for address in addresses:
                    if isinstance(address, str):
                        loc_id = self.add_location(address, [primary_id])
                
                # Map phone numbers as connection points
                phones = record.get('phones', [])
                if phones:
                    for phone in phones:
                        phone_id = self._generate_node_id({'phone': phone}, 'phone')
                        if phone_id not in self.nodes:
                            self.nodes[phone_id] = {
                                'type': 'phone',
                                'data': {'number': phone},
                                'sources': [source]
                            }
                        self.add_edge(primary_id, phone_id, 'owns_phone', weight=0.9)
                
                # Map emails
                emails = record.get('emails', []) or record.get('email', [])
                if isinstance(emails, str):
                    emails = [emails]
                    
                for email in emails:
                    if isinstance(email, str):
                        email_id = self._generate_node_id({'email': email}, 'email')
                        if email_id not in self.nodes:
                            self.nodes[email_id] = {
                                'type': 'email',
                                'data': {'address': email},
                                'sources': [source]
                            }
                        self.add_edge(primary_id, email_id, 'owns_email', weight=0.9)
                
                # Map social media
                social_media = record.get('social_media', [])
                for profile in social_media:
                    if isinstance(profile, dict):
                        platform = profile.get('platform', 'unknown')
                        url = profile.get('url', '')
                        
                        social_id = self._generate_node_id(profile, 'social')
                        if social_id not in self.nodes:
                            self.nodes[social_id] = {
                                'type': 'social_media',
                                'data': profile,
                                'sources': [source]
                            }
                        self.add_edge(primary_id, social_id, f'has_{platform.lower()}', weight=0.7)
        
        # Find cross-source connections
        self._find_cross_source_connections()
        
        return self.get_network_graph()
    
    def _find_cross_source_connections(self):
        """
        Identify and add edges between entities found in multiple sources.
        """
        logger.info("Finding cross-source connections...")
        
        # Group nodes by key identifiers
        phone_map = defaultdict(list)
        email_map = defaultdict(list)
        name_map = defaultdict(list)
        
        for node_id, node_data in self.nodes.items():
            if node_data['type'] == 'person':
                data = node_data['data']
                
                # Index by phone
                phones = data.get('phones', [])
                for phone in phones:
                    phone_map[phone].append(node_id)
                
                # Index by email
                emails = data.get('emails', []) or data.get('email', [])
                if isinstance(emails, str):
                    emails = [emails]
                for email in emails:
                    email_map[email].append(node_id)
                
                # Index by name
                name = data.get('name')
                if name:
                    name_map[name.lower()].append(node_id)
        
        # Create cross-source edges
        for phone, node_ids in phone_map.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        self.add_edge(
                            node_ids[i], node_ids[j],
                            'same_phone',
                            weight=0.95,
                            metadata={'phone': phone}
                        )
        
        for email, node_ids in email_map.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        self.add_edge(
                            node_ids[i], node_ids[j],
                            'same_email',
                            weight=0.95,
                            metadata={'email': email}
                        )
        
        for name, node_ids in name_map.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        # Lower weight for name matches (could be different people)
                        self.add_edge(
                            node_ids[i], node_ids[j],
                            'same_name',
                            weight=0.6,
                            metadata={'name': name}
                        )
    
    def get_network_graph(self) -> Dict[str, Any]:
        """
        Get complete network graph.
        
        Returns:
            Dictionary containing nodes and edges
        """
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'node_types': self._count_node_types(),
                'relationship_types': self._count_relationship_types()
            }
        }
    
    def get_subgraph(self, center_node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Extract subgraph centered on specific node.
        
        Args:
            center_node_id: Central node ID
            depth: How many hops to include
            
        Returns:
            Subgraph dictionary
        """
        if center_node_id not in self.nodes:
            return {'nodes': {}, 'edges': [], 'error': 'Node not found'}
        
        included_nodes = {center_node_id}
        current_frontier = {center_node_id}
        
        # BFS to depth
        for _ in range(depth):
            next_frontier = set()
            for node_id in current_frontier:
                # Find connected nodes
                for edge in self.edges:
                    if edge['source'] == node_id:
                        next_frontier.add(edge['target'])
                    elif edge['target'] == node_id:
                        next_frontier.add(edge['source'])
            
            included_nodes.update(next_frontier)
            current_frontier = next_frontier
        
        # Extract subgraph
        subgraph_nodes = {
            nid: self.nodes[nid] for nid in included_nodes
        }
        
        subgraph_edges = [
            edge for edge in self.edges
            if edge['source'] in included_nodes and edge['target'] in included_nodes
        ]
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'center': center_node_id,
            'depth': depth
        }
    
    def find_clusters(self, min_connections: int = 2) -> List[Set[str]]:
        """
        Find clusters of highly connected entities.
        
        Args:
            min_connections: Minimum connections to form cluster
            
        Returns:
            List of node ID sets forming clusters
        """
        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in self.edges:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
        
        # Find connected components
        visited = set()
        clusters = []
        
        def dfs(node, cluster):
            if node in visited:
                return
            visited.add(node)
            cluster.add(node)
            for neighbor in adjacency[node]:
                dfs(neighbor, cluster)
        
        for node_id in self.nodes:
            if node_id not in visited:
                cluster = set()
                dfs(node_id, cluster)
                if len(cluster) >= min_connections:
                    clusters.append(cluster)
        
        logger.info(f"Found {len(clusters)} clusters")
        return clusters
    
    def export_graph(self, format: str = 'json') -> str:
        """
        Export network graph in specified format.
        
        Args:
            format: Export format ('json', 'cytoscape', 'graphml')
            
        Returns:
            Formatted graph string
        """
        if format == 'json':
            return json.dumps(self.get_network_graph(), indent=2)
        
        elif format == 'cytoscape':
            return self._export_cytoscape()
        
        elif format == 'graphml':
            return self._export_graphml()
        
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_cytoscape(self) -> str:
        """Export in Cytoscape.js format."""
        elements = []
        
        # Add nodes
        for node_id, node_data in self.nodes.items():
            elements.append({
                'data': {
                    'id': node_id,
                    'label': self._get_node_label(node_id),
                    'type': node_data['type'],
                    **node_data['data']
                }
            })
        
        # Add edges
        for edge in self.edges:
            elements.append({
                'data': {
                    'source': edge['source'],
                    'target': edge['target'],
                    'type': edge['type'],
                    'weight': edge['weight']
                }
            })
        
        return json.dumps({'elements': elements}, indent=2)
    
    def _export_graphml(self) -> str:
        """Export in GraphML XML format."""
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
        graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        graphml.append('  <graph id="G" edgedefault="undirected">')
        
        # Nodes
        for node_id, node_data in self.nodes.items():
            label = self._get_node_label(node_id)
            graphml.append(f'    <node id="{node_id}">')
            graphml.append(f'      <data key="label">{label}</data>')
            graphml.append(f'      <data key="type">{node_data["type"]}</data>')
            graphml.append('    </node>')
        
        # Edges
        for i, edge in enumerate(self.edges):
            graphml.append(f'    <edge id="e{i}" source="{edge["source"]}" target="{edge["target"]}">')
            graphml.append(f'      <data key="type">{edge["type"]}</data>')
            graphml.append(f'      <data key="weight">{edge["weight"]}</data>')
            graphml.append('    </edge>')
        
        graphml.append('  </graph>')
        graphml.append('</graphml>')
        
        return '\n'.join(graphml)
    
    def _generate_node_id(self, data: Dict, node_type: str) -> str:
        """
        Generate unique node ID from data.
        
        Args:
            data: Node data
            node_type: Type of node
            
        Returns:
            Unique node ID
        """
        # Create deterministic ID from data
        id_string = f"{node_type}:"
        
        if 'name' in data:
            id_string += data['name']
        elif 'phone' in data:
            id_string += data['phone']
        elif 'email' in data:
            id_string += data['email']
        elif 'location' in data:
            id_string += data['location']
        elif 'address' in data:
            id_string += str(data['address'])
        else:
            id_string += str(data)
        
        # Hash to create consistent ID
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _get_node_label(self, node_id: str) -> str:
        """Get human-readable label for node."""
        node = self.nodes.get(node_id, {})
        data = node.get('data', {})
        
        if 'name' in data:
            return data['name']
        elif 'address' in data:
            return data['address']
        elif 'number' in data:
            return data['number']
        else:
            return node.get('type', 'unknown')
    
    def _merge_data(self, existing: Dict, new: Dict):
        """Merge new data into existing node data."""
        for key, value in new.items():
            if key not in existing:
                existing[key] = value
            elif isinstance(value, list):
                if isinstance(existing[key], list):
                    # Merge lists, avoid duplicates
                    existing[key] = list(set(existing[key] + value))
            elif key == 'entropy_score':
                # Take average of scores
                existing[key] = (existing[key] + value) / 2
    
    def _index_entity(self, node_id: str, data: Dict):
        """Index entity for fast lookup."""
        if 'name' in data:
            self.entity_index[data['name'].lower()].add(node_id)
        if 'phone' in data:
            self.entity_index[data['phone']].add(node_id)
        if 'email' in data:
            self.entity_index[data['email'].lower()].add(node_id)
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node['type']] += 1
        return dict(counts)
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge['type']] += 1
        return dict(counts)
