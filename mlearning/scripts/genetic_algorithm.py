"""
    Genetic Algorithm
"""
from ..unsupervised.genetic_algorithm import GeneticAlgorithm


def genetic_algr():
    """
        Runs the Genetic Algorithm model
    """
    ga = GeneticAlgorithm(target_str='Funny Human Being',
                          population=1000,
                          mutation_rate=0.05)
    ga.summarize()
    ga.run(1000, to_limit=True)
