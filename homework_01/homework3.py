#!/usr/bin/env python3

from typing import List, Tuple


class City:

  def City(self, name, x, y, z):
    pass


def read_input() -> List[City]:
  # TODO: Read input.txt from current directory
  pass


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


def main():
  pass

if __name__ == "__main__":
  main()