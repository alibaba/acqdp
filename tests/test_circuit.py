import unittest
from acqdp import circuit
import numpy as np
from scipy.stats import unitary_group
from acqdp.circuit import noise


class CircuitTestCase(unittest.TestCase):
    def test_basic(self):
        c = circuit.Circuit()
        c.append(circuit.ZeroState, [0])
        c.append(circuit.ZeroMeas, [0])
        t = c.tensor_pure
        self.assertEqual(t.contract(), 1)
        c = circuit.Circuit()
        c.append(circuit.ZeroState, [0])
        c.append(circuit.OneMeas, [0])
        t = c.tensor_pure
        self.assertEqual(t.contract(), 0)

    def test_connection(self):
        c = circuit.Circuit()
        c.append(circuit.ZeroState, [0], time_step=0)
        c.append(circuit.ZeroState, [1], time_step=0)
        with self.assertRaises(ValueError):
            c.append(circuit.ZeroMeas, [0, 1])
        c.append(circuit.ZeroState, [0], time_step=2)
        self.assertFalse(c.is_valid)
        c.append(circuit.ZeroMeas, [0], time_step=1)
        self.assertTrue(c.is_valid)
        c.append(circuit.IGate, [0], time_step=1)
        self.assertFalse(c.is_valid)
        with self.assertRaises(ValueError):
            c.tensor_density
        c.append(circuit.ZeroState, ['dummy'])
        self.assertEqual(set(c.all_qubits), {0, 1, 'dummy'})

    def random_unitary(self):
        return circuit.Unitary(1, unitary_group.rvs(2))

    def test_concatenate(self):
        u0 = self.random_unitary()
        u1 = self.random_unitary()
        s0 = circuit.ZeroState | u0
        t0 = s0.tensor_pure.contract()
        s = s0 | u1
        s_steps = circuit.PureState(1, t0) | u1
        s_assoc = circuit.ZeroState | (u0 | u1)
        t = s.tensor_pure.contract()
        t_steps = s_steps.tensor_pure.contract()
        t_assoc = s_assoc.tensor_pure.contract()
        self.assertTrue(np.allclose(t, t_steps))
        self.assertTrue(np.allclose(t, t_assoc))

    def test_parallel(self):
        u0 = self.random_unitary()
        u1 = self.random_unitary()
        u = u0 * u1
        s0 = circuit.ZeroState | u0
        s1 = circuit.ZeroState | u1
        s = (circuit.ZeroState * circuit.ZeroState) | u
        t0 = s0.tensor_pure.contract()
        t1 = s1.tensor_pure.contract()
        t = s.tensor_pure.contract()
        self.assertTrue(np.allclose(t, np.outer(t0, t1)))
        with self.assertRaises(TypeError):
            u0 * 'test'

    def test_adjoint(self):
        s = circuit.ZeroState | self.random_unitary()
        t = s.tensor_pure.contract()
        t_conj = (~s).tensor_pure.contract()
        self.assertTrue(np.allclose(t_conj, t.conj()))
        # Some operations are marked as self-adjoint, or adjoints of one another
        self.assertTrue(~(circuit.HGate) is circuit.HGate)
        self.assertTrue(~(circuit.ZGate) is circuit.ZGate)
        self.assertTrue(~(circuit.ZeroState) is circuit.ZeroMeas)
        self.assertTrue(~(circuit.ZeroMeas) is circuit.ZeroState)

    def test_density(self):
        s = circuit.ZeroState | self.random_unitary()
        t = s.tensor_pure.contract()
        t_density = s.tensor_density.contract()
        self.assertTrue(np.allclose(t_density, np.outer(t, t.conj())))
        t_density_conj = (~s).tensor_density.contract()
        self.assertTrue(np.allclose(t_density_conj, t_density.T))

    def test_dephasing(self):
        c = circuit.Dephasing(0.3)
        t = c.tensor_density.contract()
        with self.assertRaises(ValueError):
            c.tensor_pure
        with self.assertRaises(ValueError):
            c.tensor_control
        t_I = circuit.IGate.tensor_density.contract()
        t_Z = circuit.ZGate.tensor_density.contract()
        self.assertTrue(np.allclose(t, 0.7 * t_I + 0.3 * t_Z))
        s_I = circuit.ZeroState | self.random_unitary()
        s = s_I | c
        s_Z = s_I | circuit.ZGate
        t = s.tensor_density.contract()
        t_I = s_I.tensor_density.contract()
        t_Z = s_Z.tensor_density.contract()
        self.assertTrue(np.allclose(t, 0.7 * t_I + 0.3 * t_Z))
        with self.assertRaises(ValueError):
            ~s

    def test_depolariation(self):
        c = circuit.Depolarization(0.1)
        t = c.tensor_density.contract()
        t_I = circuit.IGate.tensor_density.contract()
        t_X = circuit.XGate.tensor_density.contract()
        t_Y = circuit.YGate.tensor_density.contract()
        t_Z = circuit.ZGate.tensor_density.contract()
        self.assertTrue(np.allclose(t, 0.7 * t_I + 0.1 * (t_X + t_Y + t_Z)))
        c = circuit.Depolarization(0.05, 0.1, 0.15)
        t = c.tensor_density.contract()
        self.assertTrue(np.allclose(t, 0.7 * t_I + 0.05 * t_X + 0.1 * t_Y + 0.15 * t_Z))

    def test_mixed_state(self):
        tensor = (circuit.ZeroState | circuit.Dephasing(0.3)).tensor_density
        t = tensor.contract()
        s = circuit.State(1, t)
        s1 = circuit.State(1, tensor)
        self.assertTrue(np.allclose(s.tensor_density.contract(), t))
        self.assertTrue(np.allclose(s1.tensor_density.contract(), t))
        with self.assertRaises(ValueError):
            circuit.State(2, t)
        with self.assertRaises(ValueError):
            circuit.State(2, tensor)
        with self.assertRaises(TypeError):
            circuit.State(1, 'test')

    def test_pure_state(self):
        tensor = (circuit.ZeroState | self.random_unitary()).tensor_pure
        t = tensor.contract()
        s = circuit.PureState(1, t)
        s1 = circuit.PureState(1, tensor)
        self.assertTrue(np.allclose(s.tensor_pure.contract(), t))
        self.assertTrue(np.allclose(s1.tensor_pure.contract(), t))
        with self.assertRaises(ValueError):
            circuit.PureState(2, t)
        with self.assertRaises(ValueError):
            circuit.PureState(2, tensor)
        with self.assertRaises(TypeError):
            circuit.PureState(1, 'test')

    def test_kraus(self):
        gamma = 0.3
        K0 = np.diag([1, (1 - gamma) ** 0.5])
        K1 = np.diag([0, gamma ** 0.5])[::-1]
        op = circuit.ImmutableOperation.operation_from_kraus(('io',), [K0, K1])
        t = op.tensor_density.contract()
        self.assertTrue(np.allclose(t, circuit.AmplitudeDampling(gamma).tensor_density.contract()))
        with self.assertRaises(TypeError):
            circuit.ImmutableOperation.operation_from_kraus(('io',), [K0, 'test'])
        with self.assertRaises(ValueError):
            circuit.ImmutableOperation.operation_from_kraus(('io', 'io'), [K0, K1])

    def test_controlled(self):
        CX = circuit.Controlled(circuit.XGate)
        t_CX = CX.tensor_pure.contract()
        self.assertTrue(np.allclose(t_CX, circuit.CNOTGate.tensor_pure.contract()))
        c_op = circuit.ControlledOperation(CX.tensor_control, ('c', 'io'))
        self.assertTrue(np.allclose(c_op.tensor_pure.contract(), t_CX))
        with self.assertRaises(ValueError):
            circuit.ControlledOperation(c_op.tensor_pure, ('c', 'io'))
        with self.assertRaises(ValueError):
            circuit.ControlledOperation(c_op.tensor_pure.contract(), ('c', 'io'))
        with self.assertRaises(TypeError):
            circuit.ControlledOperation('test', ('c', 'io'))
        c = circuit.Controlled(circuit.XGate, conditioned_on=False) | (circuit.IGate * circuit.XGate)
        self.assertTrue(np.allclose(c.tensor_pure.contract(), t_CX))
        cc = circuit.Controlled(c)
        CCX = circuit.Controlled(CX)
        self.assertTrue(np.allclose(cc.tensor_pure.contract(), CCX.tensor_pure.contract()))
        self.assertTrue(np.allclose(cc.tensor_pure.contract(), (~cc).tensor_pure.contract()))
        self.assertTrue(cc.is_valid)
        XX = circuit.XXRotation(1.0)
        t_XX = circuit.Controlled(XX).tensor_pure.contract()
        u_XX = circuit.Unitary(2, XX.tensor_pure)
        t_u_XX = circuit.Controlled(u_XX).tensor_pure.contract()
        self.assertTrue(np.allclose(t_XX, t_u_XX))
        with self.assertRaises(ValueError):
            circuit.Controlled(circuit.ZeroState)

    def test_superposition(self):
        u1 = circuit.CNOTGate
        u2 = self.random_unitary() * self.random_unitary()
        c = circuit.SuperPosition([u1, u2])
        t = c.tensor_pure.contract()
        t1 = u1.tensor_pure.contract()
        t2 = u2.tensor_pure.contract()
        self.assertTrue(np.allclose(t, t1 + t2))
        s0 = (circuit.ZeroState | self.random_unitary()) * (circuit.ZeroState | self.random_unitary())
        s = s0 | c
        s1 = s0 | u1
        s2 = s0 | u2
        s_sup = circuit.SuperPosition([s1, s2])
        self.assertTrue(np.allclose(s.tensor_pure.contract(), s_sup.tensor_pure.contract()))
        circuit.SuperPosition([u1, u2])
        with self.assertRaises(ValueError):
            circuit.SuperPosition([u1, u2], [0.3])
        with self.assertRaises(ValueError):
            circuit.SuperPosition([s1, u2])
        with self.assertRaises(ValueError):
            circuit.SuperPosition([circuit.IGate, circuit.Dephasing()])

    def test_rotations(self):
        c0 = circuit.XRotation(1.0) | circuit.HGate
        c1 = circuit.HGate | circuit.ZRotation(1.0)
        self.assertTrue(np.allclose(c0.tensor_pure.contract(), c1.tensor_pure.contract()))
        c0 = ~circuit.XRotation(1.0)
        c1 = circuit.XRotation(-1.0)
        self.assertTrue(np.allclose(c0.tensor_pure.contract(), c1.tensor_pure.contract()))
        c0 = ~circuit.ZRotation(1.0)
        c1 = circuit.ZRotation(-1.0)
        self.assertTrue(np.allclose(c0.tensor_pure.contract(), c1.tensor_pure.contract()))

    def test_double_rotations(self):
        c0 = circuit.XXRotation(1.0) | (circuit.HGate * circuit.HGate)
        c1 = (circuit.HGate * circuit.HGate) | circuit.ZZRotation(1.0)
        self.assertTrue(np.allclose(c0.tensor_pure.contract(), c1.tensor_pure.contract()))
        c0 = ~circuit.XXRotation(1.0)
        c1 = circuit.XXRotation(-1.0)
        self.assertTrue(np.allclose(c0.tensor_pure.contract(), c1.tensor_pure.contract()))
        c0 = ~circuit.ZZRotation(1.0)
        c1 = circuit.ZZRotation(-1.0)
        self.assertTrue(np.allclose(c0.tensor_pure.contract(), c1.tensor_pure.contract()))

    def test_unsupported_conversions(self):
        op = circuit.Operation()
        op.is_pure = True
        with self.assertRaises(ValueError):
            op.tensor_pure
        with self.assertRaises(ValueError):
            op.tensor_density

    def test_add_noise(self):
        c = circuit.Circuit()
        for i in range(3):
            c.append(circuit.HGate, [i])
        c.append(circuit.CZGate, [0, 1])
        c.append(circuit.CZGate, [0, 2])
        noise_channel = circuit.Depolarization(0.5, 0.1, 0.15)
        c_noisy = noise.add_noise(c, noise_channel)
        self.assertFalse(np.allclose(c_noisy.tensor_density.contract(), c.tensor_density.contract()))
        c_manual = circuit.Circuit()
        for i in range(3):
            c_manual.append(circuit.HGate, [i])
        for i in range(3):
            c_manual.append(noise_channel, [i])
        c_manual.append(circuit.CZGate, [0, 1])
        c_manual.append(noise_channel, [0])
        c_manual.append(circuit.CZGate, [0, 2])
        for i in range(3):
            c_manual.append(noise_channel, [i])
        self.assertTrue(np.allclose(c_noisy.tensor_density.contract(), c_manual.tensor_density.contract()))
