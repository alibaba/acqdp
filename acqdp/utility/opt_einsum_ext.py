from opt_einsum import helpers, paths
import sys
import os

try:
    from . import opt_einsum_paths, opt_einsum_helpers

    for name in opt_einsum_paths.__all__:
        setattr(paths, name, getattr(opt_einsum_paths, name))

    for name in opt_einsum_helpers.__all__:
        setattr(helpers, name, getattr(opt_einsum_helpers, name))
except ImportError:
    print('Cython modules for opt_einsum are not built. ACQDP will function normally, but contraction scheme finding may be'
          'slower. To build those modules, run:')
    print(f'    {os.path.basename(sys.executable)} -m pip install Cython')
    print(f'    {os.path.basename(sys.executable)} -m pip install --force-reinstall --no-deps acqdp')
    print('(or reinstall acqdp from whatever source you prefer)')
    print()
