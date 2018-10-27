# coding=utf-8
"""
The *sgpNet* package reconstructs Boolean network model for a gene regulatory network using single-cell data.

.. tip::
    For convenience of usage, all APIs have been imported into the top-level package *geppy* and can be used with the
    short form ``geppy.API`` directly. For example, simply use ``geppy.Chromosome`` as an alias of the class
    :class:`geppy.core.entity.Chromosome` and ``geppy.invert`` as the alias of the function
    :func:`geppy.tools.mutation.invert`.
"""


__author__ = 'Shuhua Gao'

from pkg_resources import get_distribution, DistributionNotFound

# fetch version from setup.py
try:
    __version__ = get_distribution('sgpNet').version
except DistributionNotFound as e:
    __version__ = 'Please install this package with setup.py'

from .inference import infer_Boolean_network
from .cboolnet import BooleanNetwork
