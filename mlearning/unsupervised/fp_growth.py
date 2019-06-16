"""
    Determines frequent items in a transactional dataset
"""
from dataclasses import dataclass, field

import itertools


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
    min_count:
        float = 0.333333
    tree:
        object = None
    prefixes:
        dict = {}
    frequent_items:
        list = []

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
            Gives a list of items whose occurence meets the min_count
        """
        uniques = set(itertools.chain(*db))
        transactions_count = [
            (item, self.get_transactions_count(item, db))
            for item in uniques
        ]

        fq_items = [
            transaction[0] for transaction in transactions_count
            if transactions_count[1] >= self.min_count]

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
    nodes:
        float = field(default=None)
    support:
        int = {}
    children:
        dict = {}
