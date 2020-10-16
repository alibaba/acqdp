from copy import copy, deepcopy
import numpy
from numpy import testing
import unittest
from acqdp.tensor_network import TensorValued, Tensor, TensorSum, normalize


class TensorTestCase(unittest.TestCase):

    def setUp(self):
        self.a = (2,) * 10

    def test_identifier(self):
        cnt = TensorValued.id_count
        for _ in range(100):
            a = Tensor()
        self.assertEqual(a.identifier, cnt + 100)

    def test_equal(self):
        a = Tensor(numpy.array([2, 4]))
        b = 2 * Tensor(numpy.array([1, 2]))
        c = Tensor(numpy.array([1, 1])) + Tensor(numpy.array([1, 3]))
        testing.assert_allclose(a.contract(), c.contract())
        testing.assert_allclose(a.contract(), b.contract())

    def test_data(self):
        a = numpy.random.rand(*(self.a))
        b = Tensor(a)
        testing.assert_allclose(b.contract(), a)
        self.assertEqual(self.a, b.shape)

    def test_multiplication(self):
        a = numpy.random.rand(*(self.a))
        b = Tensor(a)
        c = numpy.random.rand()
        testing.assert_allclose(c * a, (c * b).contract())

    def test_conjugation(self):
        a_R = numpy.random.rand(*(self.a))
        a_I = numpy.random.rand(*(self.a))
        a = a_R + a_I * 1j
        b = Tensor(a)
        testing.assert_allclose((~b).contract(), numpy.conj(a))
        testing.assert_allclose((~(~b)).contract(), a)

    def test_multiplication_addition(self):
        a = numpy.random.rand(10, *(self.a))
        c = numpy.random.rand(10)
        b = [c[i] * Tensor(a[i]) for i in range(10)]
        res = TensorSum()
        res_c = numpy.zeros(self.a)
        for i in range(2):
            res.add_term(i, b[i])
            res_c += c[i] * a[i]
        testing.assert_allclose(res.contract(), res_c)

    def test_transpose(self):
        a = numpy.random.rand(*(self.a))
        axes = numpy.random.permutation(10)
        b = Tensor(a) % axes
        testing.assert_allclose(b.contract(), numpy.transpose(a, axes))

    def test_mul_add_tran_conj(self):
        res_c = numpy.zeros(self.a[:2], dtype=complex)
        res = Tensor(copy(res_c))
        for _ in range(100):
            c1 = numpy.random.rand() + 1j * numpy.random.rand()
            a = numpy.random.rand(*(self.a[:2])) + 1j * \
                numpy.random.rand(*(self.a[:2]))
            c2 = numpy.random.rand() + 1j * numpy.random.rand()
            axes = numpy.random.permutation(2)
            res_c += c1 * numpy.transpose(numpy.conj(c2 * a), axes)
            res += c1 * (~(Tensor(a) * c2) % axes)
            testing.assert_allclose(res.contract(), res_c)

    def test_mul_add_tran_conj_2(self):
        res_c = numpy.zeros(self.a[:2], dtype=complex)
        res = TensorSum()
        for i in range(300):
            c1 = numpy.random.rand() + 1j * numpy.random.rand()
            a = numpy.random.rand(*(self.a[:2])) + 1j * \
                numpy.random.rand(*(self.a[:2]))
            c2 = numpy.random.rand() + 1j * numpy.random.rand()
            axes = numpy.random.permutation(2)
            res_c += c1 * numpy.transpose(numpy.conj(c2 * a), axes)
            res.add_term(i, c1 * (~(Tensor(a) * c2) % axes))
        testing.assert_allclose(res.contract(), res_c)

    def test_mul_add_tran_conj_3(self):
        res_c = numpy.zeros(self.a[:2], dtype=complex)
        res = TensorSum()
        for i in range(300):
            c1 = numpy.random.rand() + 1j * numpy.random.rand()
            a = numpy.random.rand(*(self.a[:2])) + 1j * \
                numpy.random.rand(*(self.a[:2]))
            c2 = numpy.random.rand() + 1j * numpy.random.rand()
            axes = numpy.random.permutation(2)
            res_c += c1 * numpy.transpose(numpy.conj(c2 * a), axes)
            res.add_term(i, (c1 * (~(Tensor(a) * c2) % axes)).contract())
        testing.assert_allclose(res.contract(), res_c)

    def test_mul_add_tran_conj_4(self):
        res_c = numpy.zeros(self.a[:2], dtype=complex)
        res = Tensor(copy(res_c))
        for _ in range(300):
            c1 = numpy.random.rand() + 1j * numpy.random.rand()
            a = numpy.random.rand(*(self.a[:2])) + 1j * \
                numpy.random.rand(*(self.a[:2]))
            c2 = numpy.random.rand() + 1j * numpy.random.rand()
            axes = numpy.random.permutation(2)
            res_c += c1 * numpy.transpose(numpy.conj(c2 * a), axes)
            res = Tensor(res.contract() + (c1 * (~(Tensor(a) * c2) % axes)).contract())
            testing.assert_allclose(res.contract(), res_c)

    def test_incompatible_dimensions(self):
        a = Tensor(numpy.array([1, 2]))
        b = Tensor(numpy.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            a + b

    def test_norm(self):
        a = Tensor(numpy.array([1, 2, 3]))
        e = Tensor(numpy.array([1j, 2j, 3j]))
        self.assertAlmostEqual(a.norm_squared, 14)
        self.assertAlmostEqual(e.norm_squared, 14)
        self.assertAlmostEqual((a + 2 * e).norm_squared, 70)
        c = normalize(Tensor(numpy.random.rand(*self.a)))
        self.assertAlmostEqual(c.norm_squared, 1)

    def test_copy(self):
        a = Tensor(numpy.array([1, 2]))
        b = a.copy()
        c = deepcopy(a)
        self.assertTrue(b._data is a._data)
        self.assertFalse(c._data is a._data)
        testing.assert_allclose(a.contract(), b.contract())
        testing.assert_allclose(a.contract(), c.contract())
