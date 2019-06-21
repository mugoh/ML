from ..unsupervised.fp_growth import FPGrowth

from terminaltables import AsciiTable

import numpy as np


def grow_frequent_pattern():
    """
        Creates a frequent pattern tree from
        transactions
    """
    transactions = np.array(
        ['sawdust', 'porcupines', 'butter', 'uneven'],
        ['sawdust', 'uneven', 'butter'],
        ['sawdust', 'uneven', 'marcaroni'],
        ['sawdust', 'butter', 'diapers', 'marcaroni'],
        ['sawdust', 'marcaroni', 'porcupines'])

    fp_growth = FPGrowth(min_support=3)
    print(AsciiTable([['Transactions']]).table)
    print('\n', *transactions, sep='\n')

    freq_items = fp_growth.find_frequents(transactions)
    print('\n', AsciiTable([['Frequent Items']].table))
    print(*freq_items, sep='\n')
