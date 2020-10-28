from copy import deepcopy
import numpy
import unittest
from acqdp.tensor_network import TensorNetwork


class TensorNetworkTestCase(unittest.TestCase):

    def test_initialization(self):
        a = TensorNetwork()
        a.add_node(1, [0], numpy.array([1, 2]))
        a.add_node(2, [0], numpy.array([2, 3]))
        a.open_edge(0)
        a.open_edge(1)
        a.open_edge(0)
        self.assertEqual(a.shape, (2, None, 2))
        a.add_node(3, [0, 1], numpy.array([[1, 2], [3, 4]]))
        self.assertEqual(a.shape, (2, 2, 2))

    def test_mps(self):
        c = TensorNetwork()
        for i in range(10):
            c.add_node(i, [i, (i, 'o'), (i + 1) % 10], numpy.ones((2, 3, 2)))
            c.open_edge((i, 'o'))
        self.assertEqual(c.shape, tuple([3] * 10))

        d = TensorNetwork()
        d.add_node('C', tensor=c)
        d.add_node('~C', tensor=~c)
        norm_sq = c.norm_squared
        self.assertAlmostEqual(norm_sq, d.contract())
        d.expand(['C'])
        # d.raw_data = None
        self.assertAlmostEqual(norm_sq, d.contract())
        d.expand(['~C'])
        # d.raw_data = None
        self.assertAlmostEqual(norm_sq, d.contract())
        d.encapsulate([('C', i) for i in range(10)])
        # d.raw_data = None
        self.assertAlmostEqual(norm_sq, d.contract())
        d.encapsulate([('~C', i) for i in range(10)])
        # d.raw_data = None
        self.assertAlmostEqual(norm_sq, d.contract())

    def test_update_nodes(self):
        peps = TensorNetwork()
        for i in range(3):
            for j in range(3):
                peps.add_node((i, j), [(i, j, 'h'), ((i + 1) % 3, j, 'h'), (i, j, 'v'), (i, (j + 1) % 3, 'v'), (i, j, 'o')])
                peps.open_edge((i, j, 'o'))

        self.assertEqual(peps.shape, tuple([None] * 9))

        diagonal = numpy.zeros((2, 2, 2, 2, 2))
        diagonal[0, 0, 0, 0, 0] = 1
        diagonal[1, 1, 1, 1, 1] = 1

        ready_nodes = []

        for node in peps.nodes_by_name:
            self.assertTrue(peps.is_valid)
            self.assertFalse(peps.is_ready)
            ready_nodes.append(node)
            peps.update_node(node, diagonal)

        self.assertTrue(peps.is_ready)
        self.assertTrue(peps.shape == tuple([2] * 9))

    def test_copy(self):
        a = TensorNetwork()
        a.add_node(0, [0, 1], numpy.random.rand(10, 8))
        a.add_node(1, [1, 2], numpy.random.rand(8, 9))
        a.add_node(2, [2, 0], numpy.random.rand(9, 10))
        b = a.copy()
        c = deepcopy(a)
        self.assertNotEqual(b.identifier, a.identifier)
        self.assertNotEqual(c.identifier, a.identifier)

        node = list(b.nodes_by_name)[0]
        self.assertTrue(a.nodes[(0, node)]['tensor']._data is b.nodes[(0, node)]['tensor']._data)
        self.assertFalse(a.nodes[(0, node)]['tensor']._data is c.nodes[(0, node)]['tensor']._data)

        b.update_node(0, numpy.random.rand(10, 8))
        self.assertNotEqual(a.contract(), b.contract())

    def test_shape(self):
        a = TensorNetwork()
        a.add_node(0, [0, 1], numpy.zeros((2, 3)))
        self.assertEqual((), a.shape)
        sp = []
        for _ in range(1000):
            o = numpy.random.randint(2)
            a.open_edge(o)
            sp.append(o + 2)
        self.assertEqual(tuple(sp), a.shape)

    def test_basic_contraction(self):
        a = TensorNetwork()
        a.add_node(0, [0, 1], numpy.ones((2, 2)))
        self.assertTrue(numpy.isclose(a.contract(), 4 + 0j))
        a.add_node(1, [0, 1], numpy.ones((2, 2)))
        self.assertTrue(numpy.isclose(a.contract(), 4 + 0j))
        a.update_node(1, numpy.array([[70168, 52 * 1j], [65.77, -1e-3]]))
        self.assertTrue(numpy.isclose(a.contract(), 70233.769 + 52 * 1j))

    def test_matmul(self):
        a = TensorNetwork()
        res = numpy.eye(2)
        a.open_edge(0)
        for i in range(100):
            random_mat = numpy.random.rand(2, 2)
            a.add_node(i, [i, i + 1], random_mat)
            if i > 0:
                a.close_edge(i)
            a.open_edge(i + 1)
            res = res.dot(random_mat)
        self.assertTrue(numpy.allclose(res, a.contract()))

    def test_single_tensor(self):
        a = TensorNetwork()
        a.add_node('X', [0], numpy.array([1, 0]), is_open=True)
        self.assertTrue(numpy.allclose(a.contract('khp'), [1, 0]))

    def test_transpose(self):
        a = TensorNetwork()
        b = numpy.random.rand(3, 4, 5)
        a.add_node(0, [0, 1, 2], b)
        a.open_edge(0)
        a.open_edge(1)
        a.open_edge(2)
        perm = (2, 0, 1)
        self.assertTrue(numpy.allclose((a % perm).contract(),
                                       numpy.transpose(b, perm)))

    def test_fix_index(self):
        a = TensorNetwork()
        a.add_edge(1, bond_dim=2)
        a.add_edge(2, bond_dim=2)
        a.add_edge(3, bond_dim=2)
        for i in range(10):
            a.open_edge(1)
            a.open_edge(2)
            a.open_edge(3)
        for i in range(5):
            a.fix_index(0)
            a.fix_index(10 - i)
            a.fix_index(20 - 2 * i)
        self.assertEqual(a.shape, tuple([2] * 15))
        c = numpy.zeros((2, 2, 2, 2, 2))
        c[0, 0, 0, 0, 0] = 1
        data = numpy.einsum('ABCDE, FGHIJ, KLMNO -> KAFLBCGMDHINEJO', c, c, c)
        self.assertTrue(numpy.allclose(a.contract(), data))

    def test_fix_nested_edge(self):
        a = TensorNetwork()
        a.add_edge(0, bond_dim=2)
        a.open_edge(0)
        a.open_edge(0)
        self.assertTrue(numpy.allclose(a.contract(), numpy.array([[1, 0], [0, 1]])))
        b = TensorNetwork()
        b.add_node(0, [0, 1], a)
        b.open_edge(0)
        b.open_edge(1)
        c = TensorNetwork()
        c.add_node(0, [0, 1], b)
        c.open_edge(0)
        c.open_edge(1)
        c.fix_edge(0)
        self.assertTrue(numpy.allclose(c.contract(), numpy.array([[1, 0], [0, 0]])))

    def test_fix_nested_edge_from_inside(self):
        a = TensorNetwork()
        a.add_edge(0, bond_dim=2)
        a.open_edge(0)
        a.open_edge(0)
        b = TensorNetwork()
        b.add_node(0, [0, 1], a)
        b.open_edge(0)
        b.open_edge(1)
        c = TensorNetwork()
        c.add_node(0, [0, 1], b)
        c.open_edge(0)
        c.open_edge(1)
        a.fix_edge(0)
        self.assertTrue(numpy.allclose(c.contract(), numpy.array([[1, 0], [0, 0]])))


class TensorNetworkGraphTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(517100123)
        self.a = TensorNetwork()
        self.open_edges = []
        self.closed_edges = []
        for i in range(10):
            self.a.add_edge(i, bond_dim=2)
            if numpy.random.randint(2):
                self.a.open_edge(i)
                self.open_edges.append(i)
            else:
                self.closed_edges.append(i)
        self.bipart = numpy.random.randint(2, size=(10, 10))
        for i in range(10):
            edges = numpy.where(self.bipart[i])[0]
            self.a.add_node(i,
                            edges,
                            numpy.random.rand(*([2] * len(edges))))

    def test_graph_properties(self):
        self.assertTrue(set(self.a.open_edges) == set(self.open_edges))
        self.assertTrue(self.a.closed_edges_by_name == set(self.closed_edges))

        nodes_loc = list(numpy.where(numpy.random.randint(2, size=10))[0])
        edges_from_nodes = set()
        for node in nodes_loc:
            edges_from_nodes |= set(numpy.where(self.bipart[node])[0])
        self.assertTrue(self.a.edges_by_name_from_nodes_by_name(nodes_loc) == edges_from_nodes)

        closed_edges_from_nodes = set()
        for edge in edges_from_nodes:
            if set(numpy.where(self.bipart[:, edge])[0]).issubset(nodes_loc):
                if edge not in self.open_edges:
                    closed_edges_from_nodes.add(edge)
        self.assertTrue(self.a.closed_edges_by_name_from_nodes_by_name(nodes_loc) == closed_edges_from_nodes)

        edges_loc = list(numpy.where(numpy.random.randint(2, size=10))[0])
        nodes_from_edges = set()
        for edge in edges_loc:
            nodes_from_edges |= set(numpy.where(self.bipart[:, edge])[0])
        self.assertTrue(self.a.nodes_by_name_from_edges_by_name(edges_loc) == nodes_from_edges)

    def test_graph_manipulation(self):
        data = self.a.contract()
        closed_edges = list(self.a.closed_edges_by_name)
        self.a.merge_edges(closed_edges, "merge")
        for edge in closed_edges:
            self.assertTrue((edge not in self.a.edges_by_name) or (len(self.a.network.nodes[(1, edge)]) == 0))
        for node in range(10):
            edge_list = numpy.where(self.bipart[node])[0]
            for i in range(len(edge_list)):
                if self.a.network.nodes[(0, node)]['edges'][i] == 'merge':
                    self.a.rewire(node, i, edge_list[i])

        self.assertTrue(numpy.allclose(self.a.contract(), data))
        node = numpy.random.randint(10)
        pop = self.a.pop_node(node)
        self.a.add_node(node, pop['edges'], pop['tensor'])
        self.assertTrue(numpy.allclose(self.a.contract(), data))

    def test_khp(self):
        data = self.a.contract()
        data_khp = self.a.contract('khp')
        self.assertTrue(numpy.allclose(data, data_khp))
