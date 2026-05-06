"""Genetic algorithm: genes, individuals, population lifecycle."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np

from config import DEFAULTS, IMAGE_SIZES


# ── Type definitions ──────────────────────────────────────────────────────────

Gene = Dict[str, float | int]
Individual = List[Gene]
Population = List[Individual]


# ── Gene structure ────────────────────────────────────────────────────────────

GENE_KEYS = ["x", "y", "length", "angle", "depth", "branch_factor", "color_offset", "curvature"]

GENE_BOUNDS = {
    "x": (0, None),  # Will be set based on image size
    "y": (0, None),
    "length": (40, 200),
    "angle": (-math.pi, math.pi),
    "depth": (2, 7),
    "branch_factor": (2, 5),
    "color_offset": (0.0, 1.0),
    "curvature": (-0.4, 0.4),
}


def gene_seed(gene: Gene) -> int:
    """Deterministic hash for a gene (used for stable rendering)."""
    parts = []
    for k in sorted(gene):
        v = gene[k]
        parts.append(int(v * 10000) if isinstance(v, float) else int(v))
    return hash(tuple(parts))


def random_gene(image_size: Tuple[int, int]) -> Gene:
    """Create a random gene within image-bounded constraints."""
    w, h = image_size
    return {
        "x": random.randint(w // 5, 4 * w // 5),
        "y": h - random.randint(0, h // 10),
        "length": random.randint(80, 160),
        "angle": -math.pi / 2 + random.uniform(-0.4, 0.4),
        "depth": random.randint(3, 6),
        "branch_factor": random.randint(2, 4),
        "color_offset": random.random(),
        "curvature": random.uniform(-0.25, 0.25),
    }


def create_individual(image_size: Tuple[int, int], num_fractals: int) -> Individual:
    """Create an individual with multiple fractal genes."""
    return [random_gene(image_size) for _ in range(num_fractals)]


# ── Fitness evaluation ────────────────────────────────────────────────────────

def fitness(individual: Individual) -> float:
    """Evaluate individual fitness based on visual diversity metrics.

    Considers:
    - Horizontal spread of fractal roots
    - Branch complexity (depth × branch_factor)
    - Colour variation across genes
    - Size variation for visual interest
    """
    if not individual:
        return 0.0

    xs = [g["x"] for g in individual]
    lengths = [g["length"] for g in individual]

    # Spatial spread
    spread = (max(xs) - min(xs)) if len(xs) > 1 else 0

    # Branch complexity
    branch_score = sum(g["branch_factor"] * g["depth"] for g in individual)

    # Colour diversity
    offsets = [g["color_offset"] for g in individual]
    colour_var = (max(offsets) - min(offsets)) if len(offsets) > 1 else 0

    # Size variation
    size_var = (max(lengths) - min(lengths)) if len(lengths) > 1 else 0

    # Weighted combination
    return spread * 1.0 + branch_score * 2.0 + colour_var * 100.0 + size_var * 0.5


# ── Selection ─────────────────────────────────────────────────────────────────

def select_parents(population: Population) -> Tuple[Individual, Individual]:
    """Tournament-style selection with fitness-proportionate probability."""
    if len(population) < 2:
        return population[0], population[0]

    fits = np.array([fitness(ind) for ind in population], dtype=np.float64)
    total = fits.sum()

    if total <= 0:
        probs = np.ones(len(population)) / len(population)
    else:
        # Softmax-like scaling for better selection pressure
        fits_scaled = fits - fits.min() + 1.0
        probs = fits_scaled / fits_scaled.sum()

    i1, i2 = np.random.choice(len(population), size=2, replace=False, p=probs)
    return population[i1], population[i2]


# ── Genetic operators ─────────────────────────────────────────────────────────

def mutate(individual: Individual, rate: float, image_size: Tuple[int, int]) -> Individual:
    """Apply point mutations to genes with given probability."""
    w, h = image_size
    result: Individual = []

    for gene in individual:
        ng = gene.copy()
        if random.random() < rate:
            trait = random.choice(list(gene.keys()))
            if trait == "x":
                ng[trait] = max(10, min(w - 10, gene[trait] + random.randint(-15, 15)))
            elif trait == "y":
                ng[trait] = max(10, min(h - 10, gene[trait] + random.randint(-15, 15)))
            elif trait == "depth":
                ng[trait] = max(2, min(7, gene[trait] + random.randint(-1, 1)))
            elif trait == "branch_factor":
                ng[trait] = max(2, min(5, gene[trait] + random.randint(-1, 1)))
            elif trait == "length":
                ng[trait] = max(30, min(220, gene[trait] + random.randint(-20, 20)))
            elif trait == "angle":
                ng[trait] = gene[trait] + random.uniform(-0.2, 0.2)
            elif trait == "color_offset":
                ng[trait] = max(0, min(1, gene[trait] + random.uniform(-0.15, 0.15)))
            elif trait == "curvature":
                ng[trait] = gene[trait] + random.uniform(-0.1, 0.1)
        result.append(ng)

    return result


def crossover(p1: Individual, p2: Individual) -> Individual:
    """Uniform crossover between two individuals."""
    return [
        {k: (g1[k] if random.random() < 0.5 else g2[k]) for k in g1}
        for g1, g2 in zip(p1, p2)
    ]


# ── GA Engine ─────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """Manages the full evolutionary lifecycle with smooth transitions."""

    def __init__(
        self,
        image_size: Tuple[int, int] | None = None,
        pop_size: int | None = None,
        num_fractals: int | None = None,
        mutation_rate: float | None = None,
        frames_per_gen: int | None = None,
    ):
        self.image_size = image_size or IMAGE_SIZES[DEFAULTS["image_size_key"]]
        self.pop_size = max(2, pop_size or DEFAULTS["population_size"])
        self.num_fractals = max(1, num_fractals or DEFAULTS["num_fractals"])
        self.mutation_rate = mutation_rate if mutation_rate is not None else DEFAULTS["mutation_rate"]
        self.frames_per_gen = max(1, frames_per_gen or DEFAULTS["frames_per_gen"])

        # State
        self.population: Population = []
        self.next_population: Population = []
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

        self.initialize()

    def initialize(self) -> None:
        """Create initial random population and first generation of offspring."""
        self.population = [
            create_individual(self.image_size, self.num_fractals)
            for _ in range(self.pop_size)
        ]
        self.next_population = [self._breed_child() for _ in range(self.pop_size)]
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0

        # Compute initial fitness
        fits = [fitness(ind) for ind in self.population]
        self.best_fitness = max(fits)
        self.avg_fitness = sum(fits) / len(fits)
        self.fitness_history = [self.best_fitness]
        self.avg_fitness_history = [self.avg_fitness]

    def get_interpolated_population(self) -> Population:
        """Get population with interpolated genes for smooth transitions."""
        from interpolation import interpolate_individual
        t = self.frame_idx / max(1, self.frames_per_gen)
        return [
            interpolate_individual(ind, nxt, t)
            for ind, nxt in zip(self.population, self.next_population)
        ]

    def step(self) -> bool:
        """Advance one animation frame.

        Returns
        -------
        bool
            True when a new generation is triggered.
        """
        self.frame_idx += 1
        self.global_time += 0.1

        if self.frame_idx >= self.frames_per_gen:
            self._evolve()
            return True
        return False

    def _breed_child(self) -> Individual:
        """Create offspring via selection, crossover, and mutation."""
        p1, p2 = select_parents(self.population)
        child = crossover(p1, p2)
        return mutate(child, self.mutation_rate, self.image_size)

    def _evolve(self) -> None:
        """Execute one generation: selection, elitism, breeding."""
        # Replace current population with next
        self.population = self.next_population[:]

        # Sort by fitness (descending)
        self.population.sort(key=fitness, reverse=True)

        # Update fitness records
        fits = [fitness(ind) for ind in self.population]
        self.best_fitness = fits[0]
        self.avg_fitness = sum(fits) / len(fits)
        self.fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(self.avg_fitness)

        # Elitism: keep top performers
        elite_count = max(1, self.pop_size // 3)
        elite = self.population[:elite_count]

        # Breed new population
        new_next = list(elite)
        while len(new_next) < self.pop_size:
            new_next.append(self._breed_child())

        self.next_population = new_next
        self.generation += 1
        self.frame_idx = 0

    def get_best_individual(self) -> Optional[Individual]:
        """Return the current best individual."""
        if not self.population:
            return None
        return max(self.population, key=fitness)

    def get_stats(self) -> dict:
        """Return current GA statistics."""
        return {
            "generation": self.generation,
            "frame": self.frame_idx,
            "frames_per_gen": self.frames_per_gen,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "pop_size": self.pop_size,
            "global_time": self.global_time,
        }