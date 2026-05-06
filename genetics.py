"""Genetic algorithm: genes, individuals, population lifecycle."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from config import DEFAULTS, IMAGE_SIZES


# ── Gene structure ────────────────────────────────────────────────────────────
Gene = Dict[str, float | int]


def gene_seed(gene: Gene) -> int:
    """Deterministic hash for a gene (used for stable rendering)."""
    parts = []
    for k in sorted(gene):
        v = gene[k]
        parts.append(int(v * 10000) if isinstance(v, float) else int(v))
    return hash(tuple(parts))


def random_gene(image_size: Tuple[int, int]) -> Gene:
    w, h = image_size
    return {
        "x": random.randint(w // 4, 3 * w // 4),
        "y": h,
        "length": random.randint(80, 160),
        "angle": -math.pi / 2 + random.uniform(-0.3, 0.3),
        "depth": random.randint(3, 6),
        "branch_factor": random.randint(2, 4),
        "color_offset": random.random(),
        "curvature": random.uniform(-0.2, 0.2),
    }


def create_individual(image_size: Tuple[int, int],
                      num_fractals: int) -> List[Gene]:
    return [random_gene(image_size) for _ in range(num_fractals)]


# ── Fitness ───────────────────────────────────────────────────────────────────
def fitness(individual: List[Gene]) -> float:
    xs = [g["x"] for g in individual]
    spread = (max(xs) - min(xs)) if xs else 0
    branch_score = sum(g["branch_factor"] * g["depth"] for g in individual)
    colour_var = sum(g["color_offset"] for g in individual)
    return spread + branch_score + colour_var * 50.0


# ── Selection ─────────────────────────────────────────────────────────────────
def select_parents(population: List[List[Gene]]) -> Tuple[List[Gene], List[Gene]]:
    if len(population) < 2:
        return population[0], population[0]
    fits = np.array([fitness(ind) for ind in population], dtype=np.float64)
    total = fits.sum()
    probs = (fits / total) if total > 0 else np.ones(len(population)) / len(population)
    i1, i2 = np.random.choice(len(population), size=2, replace=False, p=probs)
    return population[i1], population[i2]


# ── Genetic operators ─────────────────────────────────────────────────────────
def mutate(individual: List[Gene], rate: float,
           image_size: Tuple[int, int]) -> List[Gene]:
    result: List[Gene] = []
    for gene in individual:
        ng = gene.copy()
        if random.random() < rate:
            trait = random.choice(list(gene.keys()))
            if trait in ("x", "y"):
                ng[trait] = max(0, gene[trait] + random.randint(-10, 10))
            elif trait == "depth":
                ng[trait] = max(1, min(8, gene[trait] + random.randint(-1, 1)))
            elif trait == "branch_factor":
                ng[trait] = max(1, min(5, gene[trait] + random.randint(-1, 1)))
            elif trait == "length":
                ng[trait] = max(20, gene[trait] + random.randint(-15, 15))
            else:
                ng[trait] = gene[trait] + random.uniform(-0.1, 0.1)
        result.append(ng)
    return result


def crossover(p1: List[Gene], p2: List[Gene]) -> List[Gene]:
    return [{k: (g1[k] if random.random() < 0.5 else g2[k]) for k in g1}
            for g1, g2 in zip(p1, p2)]


# ── GA class ──────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    """Manages the full evolutionary lifecycle."""

    def __init__(
        self,
        image_size: Tuple[int, int] | None = None,
        pop_size: int | None = None,
        num_fractals: int | None = None,
        mutation_rate: float | None = None,
        frames_per_gen: int | None = None,
    ):
        self.image_size = image_size or IMAGE_SIZES[DEFAULTS["image_size_key"]]
        self.pop_size = max(1, pop_size or DEFAULTS["population_size"])
        self.num_fractals = max(1, num_fractals or DEFAULTS["num_fractals"])
        self.mutation_rate = mutation_rate if mutation_rate is not None else DEFAULTS["mutation_rate"]
        self.frames_per_gen = max(1, frames_per_gen or DEFAULTS["frames_per_gen"])

        self.population: List[List[Gene]] = []
        self.next_population: List[List[Gene]] = []
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self.best_fitness = 0.0
        self.fitness_history: List[float] = []
        self.initialize()

    # ── lifecycle ──
    def initialize(self) -> None:
        self.population = [create_individual(self.image_size, self.num_fractals)
                           for _ in range(self.pop_size)]
        self.next_population = [self._breed_child() for _ in range(self.pop_size)]
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self.best_fitness = max(fitness(ind) for ind in self.population)
        self.fitness_history = [self.best_fitness]

    def get_interpolated_population(self) -> List[List[Gene]]:
        from interpolation import interpolate_individual
        t = self.frame_idx / max(1, self.frames_per_gen)
        return [interpolate_individual(ind, nxt, t)
                for ind, nxt in zip(self.population, self.next_population)]

    def step(self) -> bool:
        """Advance one frame. Returns *True* when a new generation triggers."""
        self.frame_idx += 1
        self.global_time += 0.1
        if self.frame_idx >= self.frames_per_gen:
            self._evolve()
            return True
        return False

    # ── internal ──
    def _breed_child(self) -> List[Gene]:
        p1, p2 = select_parents(self.population)
        return mutate(crossover(p1, p2), self.mutation_rate, self.image_size)

    def _evolve(self) -> None:
        self.population = self.next_population[:]
        self.population.sort(key=fitness, reverse=True)
        self.best_fitness = fitness(self.population[0])
        self.fitness_history.append(self.best_fitness)
        elite = self.population[: max(1, self.pop_size // 3)]
        new_next = list(elite)
        while len(new_next) < self.pop_size:
            new_next.append(self._breed_child())
        self.next_population = new_next
        self.generation += 1
        self.frame_idx = 0