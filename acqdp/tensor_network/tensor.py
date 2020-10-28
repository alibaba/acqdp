import copy
import numpy
from .tensor_valued import TensorValued, DTYPE


class Tensor(TensorValued):
    """A :class:`Tensor` is an array of numbers with multiple dimensions. The most basic examples of a :class:`Tensor`
    are a vector (1-dimensional arrays of numbers) and a matrix (2-dimensional arrays).

    In our implementation, :class:`Tensor` is a subclass of :class:`TensorValued`, where the value is stored in an
    `numpy.ndarray`. All other `TensorValued` represent operations over the :class:`Tensor` objects.

    :ivar _data: `numpy.ndarray` object representing the data corresponding to the tensor.
    """

    def __init__(self,
                 data: numpy.ndarray = None,
                 dtype: type = DTYPE) -> None:
        """Constructor of a :class:`Tensor` object."""
        if data is None:
            self._data = None
        elif isinstance(data, TensorValued):
            self._data = data.contract()
        elif isinstance(data, numpy.ndarray):
            self._data = data
            if dtype is None:
                dtype = self._data.dtype
        else:
            self._data = numpy.array(data)
            if dtype is None:
                dtype = self._data.dtype
        super().__init__(dtype)

    @property
    def shape(self):
        """
        The common property of all :class:`TensorValued` classes.
        The shape of a `TensorValued` object is the bond dimension for each of its indices.
        :class:`TensorValued` objects must have compatible shapes in order to be connected together in
            a :class:`TensorNetwork`,or summed over in a :class:`TensorSum`.

        For :class:`Tensor` objects, it refers to the shape of the underlying :class:`numpy.ndarray` object.
        """
        if self._data is None:
            return None
        elif isinstance(self._data, numpy.ndarray):
            return self._data.shape
        else:
            raise ValueError

    def __str__(self) -> str:
        s = None
        if isinstance(self._data, numpy.ndarray):
            s = numpy.around(self._data, decimals=3)
        data_str = "Data: \n" + str(s)
        return super().__str__() + "\n" + data_str

    def __repr__(self) -> str:
        return "Id: " + str(self.identifier) + self.__str__()

    def __iadd__(self, t):
        self._data += t.contract()
        return self

    @property
    def is_valid(self) -> bool:
        """For :class:`Tensor` objects, it is to indicate whether the underlying :class:`numpy.ndarray` object where the
        unary operation is performed onto, is valid or not."""
        return True

    @property
    def is_ready(self) -> bool:
        """The common property of all :class:`TensorValued` classes, indicating whether the current
        :class:`TensorValued` object is ready for contraction, i.e. whether it semantically represents a tensor with a
        definite value. In the process of a program, not all :class:`TensorValued` objects need to be ready; however
        once the `data` property of a certain object is queried, such object must be ready in order to successfully
        yield an :class:`numpy.ndarray` object.

        For :class:`Tensor` objects, it is to indicate whether the underlying :class:`numpy.ndarray` object where the
        unary operation is performed onto, is ready for contraction.
        """
        return self._data is not None

    @property
    def norm_squared(self):
        """Square of Frobenius norm of the underlying :class:`numpy.ndarray` object."""
        return numpy.linalg.norm(self._data.flatten()) ** 2

    def fix_index(self, index, fix_to=0):
        """Fix the given index to the given value. The object would have the same dtype as the original one, with rank 1
        smaller than the original.

        :param index: The index to fix.
        :type index: :class:`int`.
        :param fix_to: the value to assign to the given index.
        :type fix_to: :class:`int`
        """
        if self._data is not None:
            self._data = numpy.moveaxis(self._data, index, 0)[fix_to]
        return self

    def contract(self, **kwargs):
        """
        :returns: :class:`numpy.ndarray` -- the value of the tensor whose value is stored in an :class:`numpy.ndarray` object.
        """
        return self._data

    def cast(self, dtype):
        """Return a copy of the `Tensor` object with updated underlying dtype."""
        return Tensor(numpy.array(self._data, dtype), dtype)

    def copy(self):
        """Return a copy of the `Tensor` object."""
        return Tensor(self._data, self.dtype)

    def __deepcopy__(self, memo):
        tn = Tensor(copy.deepcopy(self._data), dtype=self.dtype)
        return tn
