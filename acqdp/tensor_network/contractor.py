import inspect
import numpy
from multiprocessing import Pool
from functools import lru_cache

DEFAULT_ENGINE = 'numpy'


@lru_cache(maxsize=None)
def getDeployPlugin(pluginName):
    """plugin is just an installed python module that has "tensor_network" submodule."""
    import importlib
    res = importlib.import_module(pluginName)
    return res


class Contractor:
    """Contractor class for tensor network contraction takes a :class:`ContractionTask` object and execute it
    sequentially. For :class:`NetworkContractionTask`, multi-processing is available for further accelarate the
    computation.

    :ivar backend: When set to `jax`, large tensor contractions will make use of the `jax` backend. `numpy.einsum` is used
        otherwise.
    :ivar exeEngine: Extension interface for other contraction backends. Set to `None` by default. When set to `parallel`,
        subtasks will be computed simultaneously.
    """

    def __init__(self, exeEngine=None, backend='default', dtype=complex, **kwargs):
        self.exeEngine = exeEngine
        self.backend = backend
        self.dtype = numpy.dtype(dtype)

    def execute(self, tasks, lst=None, **kwargs):
        """Execute a contraction task.

        :param tasks: The task to be executed.
        :type tasks: :class:`acqdp.ContractionScheme`
        :param lst: The list of subtasks to be executed. If set to `None`, all subtasks are executed and merged together.
        :type lst: :class:`List`

        :returns: :class:`numpy.ndarray` -- Final result expressed as a multi-dimensional array.
        """
        tasks._load_data()
        if lst is None:
            lst = range(tasks.length)
        engine = self.exeEngine
        if engine is None:
            engine = DEFAULT_ENGINE

        if engine == 'numpy':
            return tasks._merge(
                {i: self._execute(tasks[i], **kwargs) for i in lst})
        elif inspect.ismodule(engine):
            res = engine.tensor_networkService.contractorExecute(tasks)
            return res
        elif engine == "parallel":
            with Pool() as p:
                return tasks._merge(p.starmap(self._execute, [tasks[i] for i in lst]))
        elif engine.startswith("plugin:"):
            pluginName = engine[7:]
            plugin = getDeployPlugin(pluginName)
            res = plugin.tensor_networkService.contractorExecute(tasks)
            return res

    def _execute(self,
                 task,
                 track=False,
                 normalize=False,
                 cnt=None,
                 **kwargs):
        if cnt is None:
            commands = task.commands
        else:
            commands = task.commands[:cnt]
        output = task.output
        for command in commands:
            if track:
                print("Current Memory usage = {}".format(
                    self._track_memory(commands) + 4))
            operation = command[0]
            lhs = command[1]
            rhs = command[2]
            kwargs = command[3]
            try:
                if operation == 'f':
                    res = numpy.moveaxis(lhs[0][1], kwargs['fix_idx'],
                                         range(len(kwargs['fix_idx'])))[tuple(
                                             [a[0] for a in kwargs['fix_to']])]
                    rhs[0] = (lhs[0][0], numpy.array(res))
                else:
                    if operation == 'c':
                        init_norm = sum([l[0][0] for l in lhs])
                        if 'expr' in kwargs:
                            if self.backend == 'jax':
                                res = kwargs['expr'](*[l[0][1] for l in lhs], backend='jax')
                            else:
                                res = kwargs['expr'](*[l[0][1] for l in lhs])
                        else:
                            res = numpy.array(numpy.einsum(kwargs['subscripts'],
                                                           *[l[0][1] for l in lhs]))
                    elif operation == 'n':
                        init_norm = 0
                        res = kwargs['func'](numpy.exp(lhs[0][0]) * lhs[0][1],
                                             **kwargs)
                    if normalize:
                        import numexpr as ne
                        norm = ne.evaluate(
                            'max(res.real ** 2 + res.imag ** 2)')**0.5
                        if norm == 0:
                            rhs[0] = (0, numpy.zeros(res.shape))
                        else:
                            res /= norm
                            rhs[0] = (numpy.log(norm) + init_norm, res)
                    else:
                        rhs[0] = (init_norm, res)
            except Exception as e:
                print(e)
                print(command)
                raise e
        if cnt is None:
            res = numpy.exp(output[0][0]) * output[0][1]
            return numpy.array(res)

    def _track_memory(self, commands):
        lst = []
        for command in commands:
            if command[0] == 'c':
                for k in command[1]:
                    for j in lst:
                        if k[0] is None or j is k[0][1]:
                            break
                    else:
                        lst.append(k[0][1])
            elif command[0] == 'f':
                k = command[1]
                for j in lst:
                    if k[0] is None or j is k[0][1]:
                        break
                else:
                    lst.append(k[0][1])
        return numpy.log2(max([a.size for a in lst]))


defaultContractor = Contractor()


_defaultContractor = None


def getDefault():
    global _defaultContractor
    if _defaultContractor is None:
        try:
            from acqdp.tensor_network import contractor
            _defaultContractor = contractor.defaultContractor
        except ImportError:
            _defaultContractor = Contractor()
    return _defaultContractor


def setDefault(aContractor: Contractor):
    global _defaultContractor
    _defaultContractor = aContractor


def contract(tn, **kwargs):
    return getDefault().contract(tn, **kwargs)
