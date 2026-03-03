"""
Genetic Algorithm for Fitting Bifurcation Parameters
=====================================================

This implements a simple genetic algorithm (GA) that finds the best
bifurcation parameters for the Hopf model. It works like natural selection:

    1. Start with a random population of parameter sets ("genomes")
    2. Evaluate how good each one is ("fitness" = SSIM with empirical data)
    3. Select the best ones as parents
    4. Create offspring via crossover and mutation
    5. Repeat until convergence

Key concepts:
    - Genome:    A list of 41 numbers (bifurcation parameters, one per
                 brain region in one hemisphere; mirrored for the other)
    - Fitness:   How well the model with those parameters matches real data
    - Crossover: Mixing two parent genomes to create children
    - Mutation:  Small random changes to maintain diversity
    - Elitism:   The best genomes always survive to the next generation

Dependencies: numpy
"""

import numpy as np
from random import choices, randint, randrange, random
from pathlib import Path


# ============================================================================
#  Genome operations
# ============================================================================

def generate_genome(length):
    """Create a random genome (parameter set).

    Values are drawn from a normal distribution with mean=0 and std=0.15.
    This puts most initial bifurcation parameters near zero (subcritical).

    Parameters
    ----------
    length : int
        Number of parameters (typically 41 for homotopic).

    Returns
    -------
    list of float
        Random genome.
    """
    return list(np.random.normal(loc=0, scale=0.15, size=length))


def generate_population(size, genome_length):
    """Create a population of random genomes.

    Parameters
    ----------
    size : int
        Number of genomes in the population.
    genome_length : int
        Length of each genome.

    Returns
    -------
    list of list
        Population (list of genomes).
    """
    return [generate_genome(genome_length) for _ in range(size)]


def single_point_crossover(parent1, parent2):
    """Combine two parent genomes by cutting at a random point.

    Example: parent1 = [A, B, C, D], parent2 = [1, 2, 3, 4]
    Cut at position 2 → child1 = [A, B, 3, 4], child2 = [1, 2, C, D]

    Parameters
    ----------
    parent1, parent2 : list
        Two parent genomes (same length).

    Returns
    -------
    (child1, child2) : tuple of lists
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same genome length.")
    length = len(parent1)
    if length < 2:
        return parent1, parent2
    cut = randint(1, length - 1)
    return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]


def mutate(genome, num_mutations=1, probability=0.5):
    """Apply small random changes to a genome.

    Each mutation adds Gaussian noise N(0, 0.05) to a random gene.
    Each attempt only happens with the given probability.

    Parameters
    ----------
    genome : list
        Genome to mutate (modified in place).
    num_mutations : int
        Number of mutation attempts.
    probability : float
        Probability that each attempt actually mutates.

    Returns
    -------
    list
        Mutated genome.
    """
    for _ in range(num_mutations):
        if random() < probability:
            idx = randrange(len(genome))
            genome[idx] += np.random.normal(0, 0.05)
    return genome


# ============================================================================
#  Selection and sorting
# ============================================================================

def select_parents(population, fitness_values):
    """Select two parents using fitness-proportionate (roulette wheel) selection.

    Genomes with higher fitness have a higher probability of being chosen.

    Parameters
    ----------
    population : list of list
        All genomes.
    fitness_values : list of float
        Fitness of each genome (must be non-negative!).

    Returns
    -------
    (parent1, parent2) : tuple
    """
    return tuple(choices(population=population, weights=fitness_values, k=2))


def sort_population(population, fitness_values):
    """Sort population by fitness (best first).

    Returns
    -------
    list of list
        Sorted population.
    """
    return [g for _, g in sorted(zip(fitness_values, population), reverse=True)]


# ============================================================================
#  Fitness evaluation
# ============================================================================

def evaluate_fitness(population, fitness_func):
    """Evaluate fitness for every genome in the population.

    Parameters
    ----------
    population : list of list
        All genomes.
    fitness_func : callable
        Function that takes a genome and returns a float (higher = better).

    Returns
    -------
    list of float
        Fitness values.
    """
    fitness_values = []
    for i, genome in enumerate(population):
        fitness = fitness_func(genome)
        fitness_values.append(fitness)
    print(f"  Best fitness: {max(fitness_values):.6f}")
    return fitness_values


# ============================================================================
#  Checkpoint save/load
# ============================================================================

def save_checkpoint(path, population, fitness_values, generation):
    """Save GA state to a file (so you can resume later).

    Parameters
    ----------
    path : str
        File path (will be saved as .npz).
    population : list
    fitness_values : list
    generation : int
    """
    np.savez(
        path,
        population=np.array(population, dtype=object),
        fitness_values=np.array(fitness_values),
        generation_id=generation,
    )
    print(f"  Checkpoint saved at generation {generation}")


def load_checkpoint(path):
    """Load GA state from a checkpoint file.

    Returns
    -------
    (population, fitness_values, generation) : tuple
    """
    data = np.load(path, allow_pickle=True)
    population = data["population"].tolist()
    fitness_values = data["fitness_values"].tolist()
    generation = int(data["generation_id"])
    print(f"  Checkpoint loaded: generation {generation}")
    return population, fitness_values, generation


# ============================================================================
#  Main evolution loop
# ============================================================================

def run_evolution(populate_func, fitness_func, fitness_limit=0.8,
                  generation_limit=200, convergence_limit=1e-4,
                  convergence_window=10, number_of_elites=2,
                  number_of_mutations=1, mutation_probability=0.5,
                  checkpoint_file=None):
    """Run the genetic algorithm.

    Parameters
    ----------
    populate_func : callable
        Function that returns the initial population (list of genomes).
    fitness_func : callable
        Function that scores a genome (higher = better).
    fitness_limit : float
        Stop early when best fitness reaches this threshold.
    generation_limit : int
        Maximum number of generations.
    convergence_limit : float
        Stop when fitness std drops below this (converged).
    convergence_window : int
        How many generations to check for convergence.
    number_of_elites : int
        Best genomes kept unchanged each generation.
    number_of_mutations : int
        Mutation attempts per offspring.
    mutation_probability : float
        Probability each gene actually mutates.
    checkpoint_file : str or None
        Path for checkpoint file (save & resume).

    Returns
    -------
    (population, generation, fitness_values) : tuple
        Final population, last generation number, and fitness values.
    """
    # --- Initialise or resume from checkpoint ---
    start_generation = 0
    fitness_values = []

    if checkpoint_file and Path(checkpoint_file).exists():
        population, fitness_values, start_generation = load_checkpoint(
            checkpoint_file
        )
        print(f"Resuming from generation {start_generation}")
    else:
        population = populate_func()

    print(f"GA: population={len(population)}, genome_length={len(population[0])}, "
          f"fitness_limit={fitness_limit}, max_generations={generation_limit}")

    # Track convergence
    window_values = [np.mean(fitness_values)] if fitness_values else []

    last_gen = start_generation
    for gen in range(start_generation, generation_limit):
        last_gen = gen

        # Check convergence
        if (len(window_values) >= convergence_window and
                np.std(window_values[-convergence_window:]) <= convergence_limit):
            print(f"Converged at generation {gen}!")
            break

        # Evaluate fitness
        print(f"Generation {gen}:")
        fitness_values = evaluate_fitness(population, fitness_func)

        # Sort by fitness
        sorted_pop = sort_population(population, fitness_values)
        sorted_fit = sorted(fitness_values, reverse=True)

        # Track convergence
        window_values.append(np.mean(sorted_fit))
        if len(window_values) > convergence_window:
            window_values.pop(0)

        # Print stats
        avg = sum(sorted_fit) / len(sorted_fit)
        print(f"  Gen {gen:03d} | best={sorted_fit[0]:.6f}  "
              f"avg={avg:.6f}  worst={sorted_fit[-1]:.6f}")

        # Check fitness limit
        if sorted_fit[0] >= fitness_limit:
            print(f"Fitness limit reached at generation {gen}!")
            break

        # Save checkpoint
        if checkpoint_file:
            save_checkpoint(checkpoint_file, population, fitness_values, gen)

        # Build next generation
        next_gen = sorted_pop[:number_of_elites]  # Keep the best

        while len(next_gen) < len(population):
            # Select parents
            parents = select_parents(population, fitness_values)
            # Crossover
            child1, child2 = single_point_crossover(parents[0], parents[1])
            # Mutate
            child1 = mutate(child1, number_of_mutations, mutation_probability)
            child2 = mutate(child2, number_of_mutations, mutation_probability)
            next_gen.extend([child1, child2])

        population = next_gen[:len(population)]  # Trim if odd

    return population, last_gen, fitness_values
