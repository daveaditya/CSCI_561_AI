#!/usr/bin/env python3
import sys
import functools
import random
from math import ceil, floor
from pathlib import Path
from itertools import permutations
from typing import Callable, Generator, List, Optional, Tuple
from warnings import warn

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
SelectionFunc = Callable[[Population, FitnessFunc], npt.NDArray]
CrossoverFunc = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutationFunc = Callable[[Chromosome], Chromosome]
SurvivorFunc = Callable[[Population, FitnessFunc], List[int]]
CooldownFunc = Callable[[float, float], float]
OutputFunc = Callable[[Chromosome], None]

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


def store_cities_for_path(cities: List[City], path: Chromosome, output_file_path: str) -> None:
    """Store the list of cities in the order of visit

    Args:
        cities (List[City]): List of all cities
        path (Chromosome): Path taken to visit all cities. Contains the index into the list of cities
    """
    # Create output directory if not exists
    Path("/".join(output_file_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    # Save the file
    with open(output_file_path, "w") as output:
        # print("============= FINAL OUTPUT ==============")
        for city_idx in path:
            line = f"{cities[city_idx][0]} {cities[city_idx][1]} {cities[city_idx][2]}\n"
            # print(line, end="")
            output.write(line)
        # print("=========================================")


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


def has_converged(population: Population, fitness_func: FitnessFunc) -> bool:
    """Check if all the chromosomes in the population are same or not

    Args:
        population (Population): Population to check

    Returns:
        bool: True if the population has convered, else False
    """
    fitness_score: npt.NDArray = np.apply_along_axis(fitness_func, 1, population)

    return (
        np.all(fitness_score == fitness_score[0])
        or np.array([chromosome == population[0] for chromosome in population]).all()
    )


def cooldown(temperature: float, cool_down_date: float):
    return temperature * cool_down_date


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

    # add return cost i.e. the last destination to the start city
    path_cost += distance_matrix[prev_city_idx][chromosome[0]]

    return path_cost


###########################################################################################################################
### Initial Population
###########################################################################################################################
def create_initial_population(size: int, n_allele: int, kind: str, fitness_func: FitnessFunc, **kwargs) -> Population:
    """Creates paths randomly or based on some heuristics.

    Args:
        size (int): The number of members to create for the population
        n_allele (n_allele): The number of values an allele can take

    Returns:
        Population: initial population
    """

    def generate_from_permutation(size: int, n_allele: int) -> npt.NDArray:
        """A naive method. Should not be used.
           WARN: Don't use.
        Args:
            size (int): size of the chromosome
            n_allele (int): max value of the allele

        Returns:
            npt.NDArray: First `size` number of chromosomes from permutations
        """
        warn("Do not use this method. Requires memory O(n!).", DeprecationWarning, stacklevel=2)
        return np.array([list(sample) for sample in permutations(range(n_allele))][:size])

    def generate_randomly(size, n_allele):
        random_chromosomes = list()
        rng = np.random.default_rng(seed=42)
        for _ in range(size):
            random_chromosomes.append(rng.choice(n_allele, size=n_allele, replace=False))
        return np.array(random_chromosomes)

    def generate_randomly_top(size: int, n_allele: int, fitness_func: FitnessFunc, explore_ratio: float, top_k: float):
        random_chromosomes = list()
        rng = np.random.default_rng(seed=42)
        for _ in range(ceil(size * explore_ratio)):
            random_chromosomes.append(rng.choice(n_allele, size=n_allele, replace=False))
        random_chromosomes = np.array(random_chromosomes)
        fitness_scores = np.apply_along_axis(fitness_func, 1, random_chromosomes)
        sorted_fitness_score_idxs = fitness_scores.argsort()[: ceil(size * top_k)]
        return random_chromosomes[sorted_fitness_score_idxs]

    def generate_from_cauchy_distribution(size: int, n_allele: int) -> npt.NDArray:
        random_chromosomes = list()
        rng = np.random.default_rng(seed=42)

        for _ in range(size):
            # make sure the list has unique list, because TSP cannot have duplicate cities
            while True:
                random_chromosome = np.floor(np.abs((n_allele * rng.standard_cauchy(n_allele)) % n_allele)).astype(
                    np.int32
                )
                _, unique_counts = np.unique(random_chromosome, return_counts=True)

                if all(unique_counts == 1):
                    random_chromosomes.append(random_chromosome)
                    break

        return np.array(random_chromosomes)

    if kind == "permutation":
        return generate_from_permutation(size, n_allele)
    elif kind == "random":
        return generate_randomly(size, n_allele, fitness_func)
    elif kind == "random_top":
        return generate_randomly_top(size, n_allele, fitness_func, kwargs["explore_ratio"], kwargs["top_k"])
    elif kind == "cauchy":
        return generate_from_cauchy_distribution(size, n_allele)
    else:
        raise ValueError("In valid value for kind. Must be 'random' or 'cauchy'.")


###########################################################################################################################
### Selection Methods
###########################################################################################################################
def roulette_wheel_based_selection(population: Population, fitness_func: FitnessFunc) -> Tuple[Chromosome, Chromosome]:
    """Defines the best fit individuals and selects them for breeding. Roulette wheel-based selection.

    Args:
        population (Population): list of path from which the mating pool is created

    Returns:
        npt.NDArray: a selected chromosome, ready to mate!
    """
    fitness_scores: npt.NDArray[np.float64] = np.apply_along_axis(fitness_func, 1, population)

    parents = list()

    # calculate the probablity for each chromosome, here we are minimizing so absolute of value - max fitness score
    minimize_fitness_scores = np.abs(fitness_scores - fitness_scores.max())
    population_fitness = minimize_fitness_scores.sum()

    chromosome_probability = np.array([(score / population_fitness) for score in minimize_fitness_scores])
    chromosome_probability_sorted_idx = chromosome_probability.argsort()

    minimize_fitness_scores_sorted: npt.NDArray[np.float64] = minimize_fitness_scores[
        chromosome_probability_sorted_idx[::-1]
    ]
    population_sorted: npt.NDArray = population[chromosome_probability_sorted_idx[::-1]]

    while len(parents) != 2:
        pointer = np.random.random() * population_fitness

        sum_ = 0
        for idx, score in enumerate(minimize_fitness_scores_sorted):
            sum_ += score
            if sum_ > pointer:
                parents.append(population_sorted[idx])
                break

    return tuple(parents)


def tournament_selection(population: Population, fitness_func: FitnessFunc, tournament_size: int) -> List[Chromosome]:
    """Picks `tournament_size` number of chromosomes from the population and returns the best from that batch.

    Args:
        population (Population): A collection of Chromosomes
        fitness_func (FitnessFunc): The fitness score calculating function
        tournament_size (int): The number of Chromosomes to consider in a tournament

    Returns:
       List[Chromosome]: The wining Chromosomes i.e. two parents
    """
    winners: List[Chromosome] = list()

    for _ in range(2):

        tournament_contestants_idxs: npt.NDArray = np.random.choice(range(population.shape[0]), size=tournament_size)
        tournament_contestants: Population = population[tournament_contestants_idxs]
        contestant_fitness_scores: npt.NDArray[np.float64] = np.apply_along_axis(
            fitness_func, 1, tournament_contestants
        )
        winners.append(tournament_contestants[np.argmin(contestant_fitness_scores)])

    return winners


###########################################################################################################################
### Crossover Methods
###########################################################################################################################
def ordered_crossover(
    parent_1: Chromosome, parent_2: Chromosome, crossover_probability: float
) -> Tuple[Chromosome, Chromosome]:
    """Ordered Crossover

    Args:
        parent_1 (Chromosome): list containing the random sequence of cities for the salesman to follow
        parent_2 (Chromosome): list containing the random sequence of citites for the salesman to follow

    Returns:
        Tuple[Chromosome, Chromosome]: child after performing crossover
    """
    # do mutation based on mutation probability
    do_crossover: np.float64 = np.random.rand()
    if do_crossover > crossover_probability:
        return parent_1, parent_2

    size = parent_1.shape[0]
    a, b = random.sample(range(0, size), 2)
    if a > b:
        a, b = b, a

    holes_1, holes_2 = np.full(size, True), np.full(size, True)
    for i in range(size):
        if i < a or i > b:
            holes_1[parent_2[i]] = False
            holes_2[parent_1[i]] = False

    temp_1, temp_2 = parent_1.copy(), parent_2.copy()
    k1, k2 = b + 1, b + 1

    for i in range(size):
        if not holes_1[temp_1[(i + b + 1) % size]]:
            parent_1[k1 % size] = temp_1[(i + b + 1) % size]
            k1 += 1

        if not holes_2[temp_2[(i + b + 1) % size]]:
            parent_2[k2 % size] = parent_2[(i + b + 1) % size]
            k2 += 1

    for i in range(a, b + 1):
        parent_1[i], parent_2[i] = parent_2[i], parent_1[i]

    return parent_1, parent_2


def two_point_crossover(
    parent_1: Chromosome, parent_2: Chromosome, n_allele: int, crossover_probability: float
) -> List[Chromosome]:
    # do mutation based on mutation probability
    do_crossover: np.float64 = np.random.rand()
    if do_crossover > crossover_probability:
        return parent_1, parent_2

    size: int = parent_1.shape[0]

    children: List = list()

    for _ in range(2):

        start_idx, end_idx = random.sample(range(0, size), 2)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        available_idxs: npt.NDArray = np.arange(n_allele)
        available_idxs: List[int] = np.delete(available_idxs, np.arange(start_idx, end_idx)).tolist()

        child: npt.NDArray = np.full(shape=parent_1.shape, fill_value=-1)
        child[start_idx:end_idx] = parent_1[start_idx:end_idx]

        for allele in parent_2:
            if -1 not in child:
                break
            if allele not in child:
                child[available_idxs.pop(0)] = allele

        children.append(child)

        # if len(children) == 2:
        #     if all(children[0] == children[1]):
        #         children.pop()

    return children


###########################################################################################################################
### Mutation Methods
###########################################################################################################################
def reverse_sequence_mutation(chromosome: Chromosome, mutation_probability: float) -> Chromosome:
    # do mutation based on mutation probability
    do_mutate = np.random.rand()
    if do_mutate > mutation_probability:
        return chromosome

    size = chromosome.shape[0]

    a, b = random.sample(range(1, size), 2)
    if a > b:
        a, b = b, a

    # create a copy and reverse the sub-sequence
    mutated_chromosome = chromosome.copy()
    mutated_chromosome[a:b] = chromosome[a:b][::-1]

    return mutated_chromosome


def swap_mutation(chromosome: Chromosome, mutation_probability: float) -> Chromosome:
    # do mutation based on mutation probability
    do_mutate: np.float64 = np.random.rand()
    if do_mutate > mutation_probability:
        return chromosome

    size: int = chromosome.shape[0]
    a, b = random.sample(range(1, size), 2)

    # create a copy and reverse the sub-sequence
    mutated_chromosome: Chromosome = chromosome.copy()
    mutated_chromosome[a], mutated_chromosome[b] = mutated_chromosome[b], mutated_chromosome[a]

    return mutated_chromosome


def scramble_mutation(chromosome: Chromosome, mutation_probability: float) -> Chromosome:
    # do mutation based on mutation probability
    do_mutate: np.float64 = np.random.rand()
    if do_mutate > mutation_probability:
        return chromosome

    size: Chromosome = chromosome.shape[0]
    org_sampled_idxs: npt.NDArray[np.int32] = random.sample(range(1, size), floor(size / 2))

    rng: Generator = np.random.default_rng(seed=42)
    randomized_sampled_idx: npt.NDArray[np.int32] = org_sampled_idxs.copy()
    rng.shuffle(randomized_sampled_idx)

    # create a copy and reverse the sub-sequence
    mutated_chromosome: Chromosome = chromosome.copy()
    mutated_chromosome[org_sampled_idxs] = mutated_chromosome[randomized_sampled_idx]

    return mutated_chromosome


###########################################################################################################################
### Survivor Selection Methods
###########################################################################################################################
def select_elites(fitness_scores: npt.NDArray[np.float64], elitism_rate: float) -> Tuple[int, List[int]]:
    population_size: int = fitness_scores.shape[0]

    offset = ceil(population_size * elitism_rate)
    if offset > population_size:
        raise ValueError("Elitism Rate must be between [0, 1].")

    elites = list()
    if offset:
        elites = fitness_scores.argsort()[:offset]

    return offset, elites


###########################################################################################################################
### Perform Evolution
###########################################################################################################################
def do_evolution(
    population_func: PopulationFunc,
    fitness_func: FitnessFunc,
    selection_func: SelectionFunc,
    crossover_func: CrossoverFunc,
    mutation_func: MutationFunc,
    survivor_func: SurvivorFunc,
    generation_limit: int,
    tolerance: float,
    tolerance_wait: int,
    population_decay_rate: float,
    cooldown_func: CooldownFunc,
    output_func: Optional[OutputFunc] = None,
) -> Tuple[int, Population, np.float64]:
    # initial temperature
    temperature = 10000.0

    # Generate initial population
    population: Population = population_func(fitness_func=fitness_func)
    tolerance_wait_counter: int = 0

    for gen in range(generation_limit):
        print(f"\n********************* START - GEN #{gen} *************************")
        print(f"Temperature: {temperature}")

        # if gen == 0:
        #     print("\nInitial Population: \n", population)
        # else:
        #     print("\nPopulation: \n", population)

        population_size: int = population.shape[0]
        print("Population size: ", population_size)

        # Calculate fitness score for all the chromosomes
        fitness_scores: npt.NDArray = np.apply_along_axis(fitness_func, 1, population)

        # set previous best fitness score
        fitness_scores = np.apply_along_axis(fitness_func, 1, population)
        prev_best_fitness_score = fitness_scores.min()

        # Select survivors
        survivor_offset, survivor_idxs = survivor_func(fitness_scores)

        new_population = list()

        if len(survivor_idxs) > 0:
            new_population.extend(population[survivor_idxs])
        # print("Save survivors: ", new_population)

        # reduce population each time by the `population_decay_rate`
        for _ in range(ceil(population_size * population_decay_rate) - survivor_offset):

            # select potential mates and generate children
            parent_1, parent_2 = selection_func(population, fitness_func)
            child_1, child_2 = crossover_func(parent_1, parent_2)
            # print("\nAfter Crossover: \n", new_population)

            # mutate children
            mutated_child_1, mutated_child_2 = mutation_func(child_1), mutation_func(child_2)

            new_population.extend([mutated_child_1, mutated_child_2])
            # print("\nAfter mutation: \n", new_population)

        # update population
        population = np.array(new_population)
        # print("New Population: \n", population)

        # Recalculate fitness scores
        fitness_scores = np.apply_along_axis(fitness_func, 1, population)
        fittest_chromosome_idx = fitness_scores.argmin()

        print(f"\n~~~~~~~~~~~~~~~~~~~~~ END - GEN #{gen} ~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # check for convergence
        if has_converged(population, fitness_func):
            print(f"\n!!!!!   CONVERGED -- GEN #{gen}   !!!!!\n")
            break

        # check if tolerance criteria is met
        current_best_fitness_score = fitness_scores[fittest_chromosome_idx]
        if abs(prev_best_fitness_score - current_best_fitness_score) <= tolerance:
            tolerance_wait_counter += 1
            if tolerance_wait_counter == tolerance_wait:
                print(f"\n!!!!!   TOLERANCE SATISFIED -- GEN #{gen}   !!!!\n")
                break
        prev_best_fitness_score = current_best_fitness_score

        if output_func:
            # Add the start state at the end to complete the TSP
            fittest_chromosome = population[fittest_chromosome_idx]
            fittest_chromosome = np.append(fittest_chromosome, fittest_chromosome[0])

            print("\nFitness Score: ", fitness_scores[fittest_chromosome_idx])
            output_func(path=fittest_chromosome)

        if temperature < 1000.0:
            print(f"\n!!!!!   TEMPERATURE TOO LOW -- GEN #{gen}   !!!!!\n")
            break
        temperature = cooldown_func(temperature)

    return (gen, population[fittest_chromosome_idx], fitness_scores[fittest_chromosome_idx])


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
    print("# of City : ", n_cities)
    # print("Cities: ", cities)

    # Create distance matrix
    distance_matrix = calculate_distances(n_cities, cities, euclidean)
    # print("distance matrix .. ", distance_matrix)

    n_generation, fittest_chromosome, fitness_score = do_evolution(
        # population_func=functools.partial(create_initial_population, size=3000, n_allele=n_cities, kind="random"),
        population_func=functools.partial(
            create_initial_population, size=3000, n_allele=n_cities, kind="random_top", explore_ratio=2.0, top_k=0.75
        ),
        fitness_func=functools.partial(calculate_fitness_score, distance_matrix=distance_matrix),
        # selection_func=roulette_wheel_based_selection,
        selection_func=functools.partial(tournament_selection, tournament_size=128),
        # crossover_func=functools.partial(ordered_crossover, crossover_probability=0.90),
        crossover_func=functools.partial(two_point_crossover, n_allele=n_cities, crossover_probability=0.90),
        # mutation_func=functools.partial(reverse_sequence_mutation, mutation_probability=0.30),
        mutation_func=functools.partial(swap_mutation, mutation_probability=0.25),
        # mutation_func=functools.partial(scramble_mutation, mutation_probability=0.25),
        survivor_func=functools.partial(select_elites, elitism_rate=0.2),
        generation_limit=10000,
        tolerance=1e-12,
        tolerance_wait=8,
        population_decay_rate=0.50,
        cooldown_func=functools.partial(cooldown, cool_down_date=0.90),
        output_func=functools.partial(store_cities_for_path, cities=cities, output_file_path="./output.txt"),
    )

    print("\n================   RESULTS   =====================")
    print("Generation: ", n_generation)
    print("Fitness Score: ", fitness_score)
    # print("Fittest Chromosome: ", fittest_chromosome)
    print("==================================================\n")

    # Store the final path results
    # Add the start state at the end to complete the TSP
    fittest_chromosome = np.append(fittest_chromosome, fittest_chromosome[0])
    store_cities_for_path(cities, fittest_chromosome, "./output.txt")


if __name__ == "__main__":
    main()
