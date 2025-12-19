"""
Unit tests for aggregation modules.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aggregation.aggregator import OSINTAggregator
from aggregation.network_mapper import NetworkMapper


class TestOSINTAggregator(unittest.TestCase):
    """Test OSINT aggregator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use in-memory database for testing
        # The Database class auto-initializes via _init_database() in __init__
        self.aggregator = OSINTAggregator(db_path=':memory:')
    
    def test_initialization(self):
        """Test initialization."""
        self.assertIsNotNone(self.aggregator)
        # Check that aggregator has required components
        self.assertIsNotNone(self.aggregator.db)
        self.assertIsNotNone(self.aggregator.matcher)
        self.assertIsNotNone(self.aggregator.risk_assessor)
    
    def test_initialize_scrapers(self):
        """Test scraper initialization."""
        scrapers = self.aggregator._initialize_scrapers()
        self.assertIsInstance(scrapers, dict)
        # Should have at least the scrapers we created
        self.assertGreater(len(scrapers), 0)
    
    def test_has_required_methods(self):
        """Test that aggregator has required methods."""
        self.assertTrue(hasattr(self.aggregator, 'search_person'))
        self.assertTrue(hasattr(self.aggregator, '_aggregate_results'))
        self.assertTrue(hasattr(self.aggregator, '_find_or_create_record'))
        self.assertTrue(hasattr(self.aggregator, '_merge_result_into_record'))


class TestNetworkMapper(unittest.TestCase):
    """Test network mapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapper = NetworkMapper()
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(len(self.mapper.nodes), 0)
        self.assertEqual(len(self.mapper.edges), 0)
    
    def test_add_person(self):
        """Test adding person node."""
        person_data = {
            'name': 'John Doe',
            'age': 35,
            'phones': ['555-1234']
        }
        
        node_id = self.mapper.add_person(person_data, 'test_source')
        self.assertIsNotNone(node_id)
        self.assertIn(node_id, self.mapper.nodes)
        self.assertEqual(self.mapper.nodes[node_id]['type'], 'person')
    
    def test_add_edge(self):
        """Test adding relationship edge."""
        person1 = self.mapper.add_person({'name': 'Person 1'}, 'source1')
        person2 = self.mapper.add_person({'name': 'Person 2'}, 'source2')
        
        self.mapper.add_edge(person1, person2, 'relative', weight=0.8)
        
        self.assertEqual(len(self.mapper.edges), 1)
        self.assertEqual(self.mapper.edges[0]['type'], 'relative')
        self.assertEqual(self.mapper.edges[0]['weight'], 0.8)
    
    def test_get_network_graph(self):
        """Test network graph export."""
        # Add some test data
        self.mapper.add_person({'name': 'Test Person'}, 'test')
        
        graph = self.mapper.get_network_graph()
        
        self.assertIn('nodes', graph)
        self.assertIn('edges', graph)
        self.assertIn('stats', graph)
        self.assertEqual(graph['stats']['node_count'], 1)
    
    def test_map_relationships_empty(self):
        """Test mapping with empty data."""
        empty_data = {
            'sources': {}
        }
        
        result = self.mapper.map_relationships(empty_data)
        self.assertIn('nodes', result)
        self.assertIn('edges', result)
        self.assertEqual(len(result['nodes']), 0)
    
    def test_map_relationships_with_data(self):
        """Test mapping with actual data."""
        test_data = {
            'sources': {
                'test_source': {
                    'records': [
                        {
                            'name': 'John Doe',
                            'age': 35,
                            'phones': ['555-1234'],
                            'relatives': ['Jane Doe', 'Bob Doe']
                        }
                    ]
                }
            }
        }
        
        result = self.mapper.map_relationships(test_data)
        
        self.assertIn('nodes', result)
        self.assertIn('edges', result)
        # Should have created nodes for person and relatives
        self.assertGreater(len(result['nodes']), 0)
    
    def test_find_clusters_empty(self):
        """Test cluster finding with no data."""
        clusters = self.mapper.find_clusters()
        self.assertIsInstance(clusters, list)
        self.assertEqual(len(clusters), 0)
    
    def test_find_clusters_with_data(self):
        """Test cluster finding with connected nodes."""
        # Create a small network
        p1 = self.mapper.add_person({'name': 'Person 1'}, 'test')
        p2 = self.mapper.add_person({'name': 'Person 2'}, 'test')
        p3 = self.mapper.add_person({'name': 'Person 3'}, 'test')
        
        self.mapper.add_edge(p1, p2, 'relative')
        self.mapper.add_edge(p2, p3, 'relative')
        
        clusters = self.mapper.find_clusters(min_connections=2)
        
        self.assertIsInstance(clusters, list)
        # Should find at least one cluster
        if clusters:
            self.assertGreater(len(clusters[0]), 0)
    
    def test_export_graph_json(self):
        """Test JSON export."""
        # Add some test data
        self.mapper.add_person({'name': 'Test User'}, 'test')
        
        json_output = self.mapper.export_graph(format='json')
        self.assertIsInstance(json_output, str)
        self.assertIn('nodes', json_output)
        self.assertIn('edges', json_output)
    
    def test_export_graph_cytoscape(self):
        """Test Cytoscape export."""
        self.mapper.add_person({'name': 'Test User'}, 'test')
        
        cyto_output = self.mapper.export_graph(format='cytoscape')
        self.assertIsInstance(cyto_output, str)
        self.assertIn('elements', cyto_output)
    
    def test_add_location(self):
        """Test adding location node."""
        person_id = self.mapper.add_person({'name': 'Test User'}, 'test')
        
        loc_id = self.mapper.add_location('123 Test St, Miami FL', [person_id])
        
        self.assertIsNotNone(loc_id)
        self.assertIn(loc_id, self.mapper.nodes)
        self.assertEqual(self.mapper.nodes[loc_id]['type'], 'location')
        # Should have created edge
        self.assertGreater(len(self.mapper.edges), 0)
    
    def test_subgraph_extraction(self):
        """Test subgraph extraction."""
        person_id = self.mapper.add_person({'name': 'Center Person'}, 'test')
        
        subgraph = self.mapper.get_subgraph(person_id, depth=1)
        
        self.assertIn('nodes', subgraph)
        self.assertIn('edges', subgraph)
        self.assertIn('center', subgraph)
        self.assertEqual(subgraph['center'], person_id)
        self.assertIn(person_id, subgraph['nodes'])
    
    def test_subgraph_with_connections(self):
        """Test subgraph with connected nodes."""
        p1 = self.mapper.add_person({'name': 'Center'}, 'test')
        p2 = self.mapper.add_person({'name': 'Connected'}, 'test')
        
        self.mapper.add_edge(p1, p2, 'relative')
        
        subgraph = self.mapper.get_subgraph(p1, depth=1)
        
        # Should include both nodes
        self.assertEqual(len(subgraph['nodes']), 2)
        self.assertEqual(len(subgraph['edges']), 1)
    
    def test_add_organization(self):
        """Test adding organization node."""
        org_id = self.mapper.add_organization('Test Corp', {'industry': 'Tech'})
        
        self.assertIsNotNone(org_id)
        self.assertIn(org_id, self.mapper.nodes)
        self.assertEqual(self.mapper.nodes[org_id]['type'], 'organization')
    
    def test_cross_source_connections(self):
        """Test finding cross-source connections."""
        # Add same person from different sources with shared phone
        data = {
            'sources': {
                'source1': {
                    'records': [
                        {'name': 'John Doe', 'phones': ['555-1234']}
                    ]
                },
                'source2': {
                    'records': [
                        {'name': 'John Doe', 'phones': ['555-1234']}
                    ]
                }
            }
        }
        
        result = self.mapper.map_relationships(data)
        
        # Should create cross-source connections
        self.assertGreater(len(result['edges']), 0)


if __name__ == '__main__':
    unittest.main()
