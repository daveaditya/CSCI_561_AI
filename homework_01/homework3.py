#!/usr/bin/env python3
import sys
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import euclidean

City = Tuple[int, int, int]

def read_input(input_file_path: str) -> Tuple[int, List[City]]:
    # TODO: Read input.txt from current directory
    cities: List[City] = list()

    with open(input_file_path, "r") as input_file:
        city_count = int(input_file.readline())
        for line in input_file.readlines():
            cities.append(tuple(map(int, line.split(sep=" "))))

    return city_count, cities


def calculate_distances(n_cities: int, cities: City) -> npt.ArrayLike:
    distance_matrix = np.empty([n_cities, n_cities])
    for start_city_idx in range(n_cities):
        for end_city_idx in range(n_cities):
            distance_matrix[start_city_idx][end_city_idx] = euclidean(cities[start_city_idx], cities[end_city_idx])
    return distance_matrix


def store_output(path: List[City]):
    # TODO: Store output in given format to a output.txt file in current directory
    pass


# a. initial population
def create_initial_population(size: int, cities: List[int]):
    """Creates paths randomly or based on some heuristics.

    Args:
        size (int): _description_
        cities (List[int]): _description_

    Returns:
        List[int]: initial population
    """
    initial_population: List[int] = list()

    # TODO: logic here ...

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
    distance_matrix = calculate_distances(n_cities, cities)
    print("distance matrix .. ", distance_matrix)

    pass


if __name__ == "__main__":
    main()
