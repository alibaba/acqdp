from copy import deepcopy
import numpy
from numpy import testing
import unittest
from acqdp.tensor_network import TensorSum, Tensor, TensorNetwork


class TensorSumTestCase(unittest.TestCase):

    def setUp(self):
        self.a = TensorNetwork()
        for i in range(5):
            self.a.open_edge(i)
        for i in range(5):
            for j in range(5, 10):
                self.a.add_node((i, j), [i, j], numpy.random.rand(2, 2))

        self.b = TensorNetwork()
        for i in range(5):
            self.b.open_edge(i)
        for i in range(5):
            for j in range(5, 10):
                self.b.add_node((i, j), [i, j], 1j * numpy.random.rand(2, 3))

        self.c = Tensor(numpy.random.rand(2, 2, 2, 2, 2))

    def test_addition(self):
        c = self.a + ~(self.b) + 8 * self.c
        self.assertEqual(type(c), TensorSum)
        self.assertTrue(numpy.allclose(c.contract(),
                                       self.a.contract()
                                       + numpy.conj(self.b.contract())
                                       + 8 * self.c.contract()))

    def test_associativity(self):
        c = (self.a + self.b) + self.c
        d = self.a + (self.b + self.c)
        e = (self.a + (self.c - self.b)) + (self.a + (2 * self.b - self.a))
        self.assertTrue(numpy.allclose(c.contract(), d.contract()))
        self.assertTrue(numpy.allclose(c.contract(), e.contract()))

    def test_transpose(self):
        axesA = (2, 4, 0, 1, 3)
        axesB = (3, 4, 1, 0, 2)
        axesC = (3, 0, 1, 2, 4)
        axes_ = (1, 3, 0, 4, 2)
        c = (self.a % axesA
             + self.b % axesB
             + self.c % axesC) % axes_
        data = numpy.transpose(numpy.transpose(self.a.contract(), axesA)
                               + numpy.transpose(self.b.contract(), axesB)
                               + numpy.transpose(self.c.contract(), axesC), axes_)
        self.assertTrue(numpy.allclose(c.contract(), data))

    def test_add_and_remove_term(self):
        c = TensorSum()
        for i in range(100):
            c.add_term(i, self.a % numpy.random.permutation(5))
        choice = numpy.random.choice(100, 30, replace=False)
        for j in choice:
            c.remove_term(j)
        with self.assertRaises(KeyError):
            c.remove_term(choice[0])
        with self.assertRaises(KeyError):
            c.add_term(0, None)
            c.add_term(0, None)

    def test_shape_cache(self):
        c = TensorSum()
        misshaped = numpy.random.rand(2, 2, 3, 2, 2)
        c.add_term(0, self.a)
        c.add_term(1, self.a)
        c.remove_term(0)
        with self.assertRaises(ValueError):
            c.add_term(2, misshaped)
        c.add_term(2, self.a)
        c.remove_term(1)
        c.remove_term(2)
        # Now that C is empty we can add any shape to it
        c.add_term(0, misshaped)
        c.add_term(1, misshaped)
        with self.assertRaises(ValueError):
            c.add_term(2, self.a)
        # The function update_term does not fail early, so the user can update all tensors one-by-one
        c.update_term(0, self.a)
        c.update_term(1, self.a)
        c.add_term(2, self.a)

    def test_copy(self):
        c = TensorSum()
        a = Tensor(numpy.random.rand(2, 2, 2, 2, 2))
        for i in range(2):
            c.add_term(i, (a % numpy.random.permutation(5)).expand())
        d = c.copy()
        testing.assert_allclose(c.contract(), d.contract())
        e = deepcopy(c)
        testing.assert_allclose(c.contract(), e.contract())
        d.update_term(0, numpy.random.rand(2, 2, 2, 2, 2))
        self.assertFalse(numpy.allclose(c.contract(), d.contract()))
