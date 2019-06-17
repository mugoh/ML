"""
    Data Mining: Determines frequent items in a transactional dataset
"""
from dataclasses import dataclass, field

from typing import Any
import itertools
import operator


@dataclass
class FPGrowth:
    """
        Builds an FP Growth tree, minable for collection
        of frequent dataset items.

        Parameters
        ----------
        min_support: float
            Minimum transaction ratio to deem an item as frequent
    """
    min_support: float = 0.333333
    tree: object = None
    prefixes: dict = {}
    frequent_items: list = []

    def find_frequents(self, transactions, suffix=None):
        """
            Gets frequent items sets from transactions
        """

        self.transactions = transactions
        self.determine_frequent_item_sets(transactions, suffix=suffix)

    def determine_frequent_item_sets(self, conditional_db, suffix=None):
        """
            Finds and updates frequent items from the conditional database
        """
        freq_items = self.__get_frequents_list(self)

    def __get_frequents_list(self, db: 'list') -> list:
        """
            Gives a list of items whose occurence meets the min_support
        """
        uniques = set(itertools.chain(*db))
        transactions_count = [
            (item, self.get_transactions_count(item, db))
            for item in uniques
        ]

        fq_items = [
            transaction for transaction in transactions_count
            if transactions_count[1] >= self.min_support]

        fq_items = [fq_item[0] for fq_item in sorted(
            fq_items,
            key=operator.itemgetter(1),
            reverse=True)
        ]

        return fq_items

    def get_transactions_count(self, item, trans):
        """
            Finds the number of transactions in which an item is
            present
        """
        holding_transactions = [count for transaction in trans
                                for count in transaction if item in transaction
                                ]

        return len(holding_transactions)

    def create_tree(self, transactions, fq_items):
        """
            Creates the F Pattern growth tree
        """
        self.root = FPTreeNode()

        for transaction in transactions:
            transac = [item for item in transaction if item in fq_items]
            transac.sort(key=operator.itemgetter(fq_items))
            self.insert_node(self.root, transac)


def insert_node(self, parent, nodes):
    """
        Inserts nodes to tree
    """
    child_ = nodes[0]
    child = FPTreeNode(item=child_)

    if child_ in parent.children:
        child_.children[child.node].support += 1
    else:
        child_.chidren[child.node] = child
        self.insert_node(parent.children[child.node], nodes[1:])


@dataclass
class FPTreeNode:
    """
        FP Tree item

        Parameters
        ----------
        node: float
            Value of the node
        support: float
            No. of occurrence in the transaction
        children: dict
            Child nodes in the growth tree
    """
    node: Any = field(default=None)
    support: int = 1
    children: dict = {}
