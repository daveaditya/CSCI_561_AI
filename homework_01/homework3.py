#!/usr/bin/env python3
import sys
import functools
from math import factorial
from itertools import permutations
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import euclidean

# Type Hints
City = Tuple[int, int, int]

Chromosome = List[int]  # represents one solution i.e. a single path in TSP
Population = List[Chromosome]  # represents all the paths
PopulationFunc = Callable[[], Population]  # A function that generates the population
FitnessFunc = Callable[
    [Chromosome, npt.NDArray[np.float64]], np.float64
]  # A function that takes a chromosome and distance matrix to give fitness score of the chromosome
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Chromosome, Chromosome]]
CrossoverFunc = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutationFunc = Callable[[Chromosome], Chromosome]


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
        path (Genome): Path taken to visit all cities. Contains the index into the list of cities
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


# a. initial population
def create_initial_population(size: int, n_allele: int) -> Population:
    """Creates paths randomly or based on some heuristics.

    Args:
        size (int): The number of members to create for the population
        n_allele (n_allele): The number of values an allele can take

    Returns:
        Population: initial population
    """
    # a.1 Generate Randome Path / Genome
    def generate_random_chromosome(size: int, n_allele: int):
        random_chromosomes = list(permutations(range(n_allele)))[:size]
        return list(map(lambda t: [*t, t[0]], random_chromosomes))

    initial_population: Population = generate_random_chromosome(size, n_allele)
    return initial_population


# b. parent selection
def create_mating_pool(population: List[List[int]], rank_list: Tuple[int, float]):
    """Defines the best fit individuals and selects them for breeding. Roulette wheel-based selection.

    Args:
        population (List[List[int]]): list of path from which the mating pool is created
        rank_list (Tuple[int, float]): tuple of index and fitness scores sorted in descending order

    Returns:
        List[int]: list of populations selected for mating
    """
    mating_pool: List[int] = list()

    # TODO: logic here ...

    return mating_pool


# c. crossover
def crossover(parent_1: List[int], parent_2: List[int], start_index: int, end_index: int):
    """two point crossover

    Args:
        parent_1 (List[int]): list containing the random sequence of cities for the salesman to follow
        parent_2 (List[int]): list containing the random sequence of citites for the salesman to follow
        start_index (int): start index of the subarray to be chose from parent 1
        end_index (int): end index of the subarray to be chosen from parent 1

    Returns:
        List[int]: child after performing crossover
    """
    child: List[int] = list()

    return child


# d. mutation
# e. find and store best

# f. run evolution

# TODO: note, minimize distance


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

    # Generate initial population
    initial_population: Population = create_initial_population(size=4, n_allele=n_cities)
    print("initial population ...", initial_population)

    # Calculate fitness score for all the chromosomes
    print(
        "fitness scores ... ",
        list(map(functools.partial(calculate_fitness_score, distance_matrix=distance_matrix), initial_population)),
    )


if __name__ == "__main__":
    main()
