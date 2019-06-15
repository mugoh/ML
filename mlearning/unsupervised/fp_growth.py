"""
    Determines frequent items in a transactional dataset
"""
from dataclasses import dataclass, field


@dataclass
class FPGrowth:
    """
        Builds an FP Growth tree, minable for collection
        of frequent dataset items.

        Parameters
        ----------
        min_count: float
            Minimum transaction ratio to deem an item as frequent
    """
    min_count: float = 0.333333
    tree: object = None
    prefixes: dict = {}
    frequent_items: list = []


@dataclass
class FPTreeNode:
    """
        FP Tree item

        Parameters
        ----------
        node: float
            Value of the node
        frequency: float
            No. of occurrence in the transaction
        children: dict
            Child nodes in the growth tree
    """
    nodes: float = field(default=None)
    support: int = {}
    children: dict = {}
