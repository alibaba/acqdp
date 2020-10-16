.. _installation:

=================
Overview
=================

Alibaba-Cloud Quantum Development Platform (ACQDP) is a quantum computing framework written in Python with the aim of realizing the potential of quantum computing in research, education and business.



=================
Installation
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

Prerequisites
***************************

`Python <https://docs.python-guide.org/>`__ - version 3.7 or later is required. 

`Cython <https://cython.org>`__ (optional) - used in the acqdp to accelerate the software package `KaHyPar <https://github.com/kahypar>`__ used in contraction order finding. 
In a command window, run 

.. code-block:: bash

   pip install cython

Installation from PyPI
**************************

ACQDP packages are published on the `Python Package Index <https://pypi.org/project/ACQDP/>`__ and can be installed using `pip`.
This is the recommended way for most users. However, if you'd like to see or modify the source code, proceed to the next section 
`Installation from source code`_

In a command window, run 

.. code-block:: bash

   pip install -U acqdp

Installation from source code
*****************************

Alternative, you can install from source code, say, cloned from `Github <https://github.com/alibaba/acqdp>`__ so you can see and modify the code. 
First clone the repo

.. code-block:: bash

   git clone https://github.com/alibaba/acqdp


Then install the ACQDP packages. Enter the `acqdp` folder and run

.. code-block:: bash

   pip install -e .

Installation Issues
****************************
If you are installing on Windows, you may encounter an error when one of its dependencies, 
`kahypar <https://pypi.org/project/kahypar>`__, is being installed. This is a limitation of kahypar.
You may try to build it from its source code follow instructions 
`here <https://github.com/kahypar/kahypar#requirements>`__


Example and test codes
***********************

You are ready to go! Find some examples at https://github.com/alibaba/acqdp/tree/master/examples

To run the example in the examples folder examples/foo.py, run the following command in the home folder:

.. code-block:: bash

   python examples/foo.py

For example, we can run the script `ghz.py` in the examples folderby the following code:

.. code-block:: bash

   python examples/ghz.py


To see if the package passes all the tests, simply run :command:`python -m pytest`.
