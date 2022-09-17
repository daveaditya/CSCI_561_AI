#!/usr/bin/env python3
import sys
import functools
from math import factorial
from itertools import permutations
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import euclidean

###########################################################################################################################
### Type Hinting
###########################################################################################################################
City = Tuple[int, int, int]

Chromosome = npt.NDArray[np.int32]  # represents one solution i.e. a single path in TSP
Population = npt.NDArray[np.int32]  # represents all the paths
PopulationFunc = Callable[[], Population]  # A function that generates the population
FitnessFunc = Callable[
    [Chromosome, npt.NDArray[np.float64]], np.float64
]  # A function that takes a chromosome and distance matrix to give fitness score of the chromosome
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Chromosome, Chromosome]]
CrossoverFunc = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutationFunc = Callable[[Chromosome], Chromosome]


###########################################################################################################################
### Helper Functions
###########################################################################################################################
def read_input(input_file_path: str) -> Tuple[int, List[City]]:
    """Parses the input file and returns the number of cities and a list containing the coordinates of the cities.

    Args:
        input_file_path (str): Relative or absolute path to the input file

    Returns:
        Tuple[int, List[City]]: Number of Cities and a list of city coordinates
    """
    cities: List[City] = list()

    with open(input_file_path, "r") as input_file:
        city_count = int(input_file.readline())
        for line in input_file.readlines():
            cities.append(tuple(map(int, line.split(sep=" "))))

    return city_count, cities


def print_cities_for_path(cities: List[City], path: Chromosome) -> None:
    """Print the list of cities in the order of visit

    Args:
        cities (List[City]): List of all cities
        path (Chromosome): Path taken to visit all cities. Contains the index into the list of cities
    """
    for city_idx in path:
        print(f"{cities[city_idx][0]}{cities[city_idx][1]}{cities[city_idx][2]}")


def calculate_distances(n_cities: int, cities: List[City], distance_func: Callable) -> npt.NDArray[np.float64]:
    """Calculates the disance between each city.

    Args:
        n_cities (int): Number of cities
        cities (List[City]): List of cities
        distance_func (Callable): The distance function to use

    Returns:
        npt.ArrayLike: a n_cities x n_cities array containing the distance between each city
    """
    distance_matrix = np.empty([n_cities, n_cities])
    for start_city_idx in range(n_cities):
        for end_city_idx in range(n_cities):
            distance_matrix[start_city_idx][end_city_idx] = distance_func(cities[start_city_idx], cities[end_city_idx])
    return distance_matrix


###########################################################################################################################
### Fitness Score Calculator
###########################################################################################################################
def calculate_fitness_score(chromosome: Chromosome, distance_matrix: npt.NDArray[np.float64]) -> np.float64:
    """Calculates the fitness score of a Chromosome using the distance matrix.

    Args:
        chromosome (Chromosome): A possible path
        distance_matrix (npt.NDArray[np.float64]): A matrix with call the distances for each city

    Returns:
        np.float64: Total cost / distance for the chromosome (path)
    """
    path_cost = 0.0
    prev_city_idx = chromosome[0]
    for idx in chromosome:
        path_cost += distance_matrix[prev_city_idx][idx]
        prev_city_idx = idx

    return path_cost


###########################################################################################################################
### Initial Population
###########################################################################################################################
def create_initial_population(size: int, n_allele: int) -> Population:
    """Creates paths randomly or based on some heuristics.

    Args:
        size (int): The number of members to create for the population
        n_allele (n_allele): The number of values an allele can take

    Returns:
        Population: initial population
    """
    # a.1 Generate Randome Path / Chromosome
    def generate_random_chromosome(size: int, n_allele: int):
        random_chromosomes = list(permutations(range(n_allele)))[:size]
        return np.fromiter(map(lambda t: [*t, t[0]], random_chromosomes))

    initial_population: Population = generate_random_chromosome(size, n_allele)
    return initial_population


###########################################################################################################################
### Selection Methods
###########################################################################################################################
def roulette_wheel_based_selection(population: Population, fitness_func: FitnessFunc) -> Chromosome:
    """Defines the best fit individuals and selects them for breeding. Roulette wheel-based selection.

    Args:
        population (Population): list of path from which the mating pool is created

    Returns:
        Chromosome: a selected chromosome, ready to mate!
    """
    fitness_scores: npt.NDArray[np.float64] = np.fromiter(map(fitness_func, population))
    population_fitness = fitness_scores.sum()

    # calculate the probablity for each chromosome, here we are minimizing and hence 1 - probability
    chromosome_probability = [(1 - score / population_fitness) for score in fitness_scores]

    return np.random.choice(population, chromosome_probability)


###########################################################################################################################
### Crossover Methods
###########################################################################################################################
def crossover(parent_1: Chromosome, parent_2: Chromosome, start_index: int, end_index: int) -> Chromosome:
    """two point crossover

    Args:
        parent_1 (Chromosome): list containing the random sequence of cities for the salesman to follow
        parent_2 (Chromosome): list containing the random sequence of citites for the salesman to follow
        start_index (int): start index of the subarray to be chose from parent 1
        end_index (int): end index of the subarray to be chosen from parent 1

    Returns:
        Chromosome: child after performing crossover
    """
    child: Chromosome = list()

    return child


###########################################################################################################################
### Mutation Methods
###########################################################################################################################


# e. find and store best


###########################################################################################################################
### Execute Evolution
###########################################################################################################################
def execute_evolution(
    population_func: PopulationFunc,
    fitness_func: FitnessFunc,
    selection_func: Optional[SelectionFunc],
    crossover_func: Optional[CrossoverFunc],
    mutation_func: Optional[MutationFunc],
    generation_limit: int,
) -> Tuple[Population, np.float64]:
    # Generate initial population
    initial_population: Population = population_func()
    print("initial population ...", initial_population)

    # Calculate fitness score for all the chromosomes
    print(
        "fitness scores ... ",
        fitness_func(initial_population),
    )
    pass


###########################################################################################################################
### Main
###########################################################################################################################
def main():
    # Take test files as input from commandline or use the default "input.txt" file
    input_file_path = "./input.txt"
    if len(sys.argv) == 2:
        input_file_path = sys.argv[1]

    # Read input file
    n_cities, cities = read_input(input_file_path)
    print("n_cities  ... ", n_cities)
    print("cities ... ", cities)

    # Create distance matrix
    distance_matrix = calculate_distances(n_cities, cities, euclidean)
    print("distance matrix .. ", distance_matrix)

    last_population, generation = execute_evolution(
        population_func=functools.partial(create_initial_population, size=4, n_allele=n_cities),
        fitness_func=functools.partial(calculate_fitness_score, distance_matrix=distance_matrix),
        selection_func=None,
        crossover_func=None,
        mutation_func=None,
        generation_limit=100,
    )

    print("Generation: ", generation)
    print("last_population ... ", last_population)


if __name__ == "__main__":
    main()
