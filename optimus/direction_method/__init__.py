"""Implementation of methods for choosing a step direction when optimizing.

Due to implementation details, the direction returned by these methods should
be the _opposite_ of the actual step taken, as it's subtracted from the previous parameters.

In order to implement a custom method for selecting a direction, inherit from
`optimus.direction_method.DirectionMethod`.
"""
from .direction_method import DirectionMethod
