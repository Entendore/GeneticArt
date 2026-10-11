"""Genetic algorithm: genes, individuals, population lifecycle with save/load."""

from __future__ import annotations

import json
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np

from config import DEFAULTS, IMAGE_SIZES


Gene = Dict[str, float | int]
Individual = List[Gene]
Population = List[Individual]

GENE_KEYS = [
    "x", "y", "length", "angle", "depth",
    "branch_factor", "color_offset", "curvature",
]


def gene_seed(gene: Gene) -> int:
    parts = []
    for k in sorted(gene):
        v = gene[k]
        parts.append(int(v * 10000) if isinstance(v, float) else int(v))
    return hash(tuple(parts))


def random_gene(image_size: Tuple[int, int], rng: random.Random) -> Gene:
    w, h = image_size
    return {
        "x": rng.randint(w // 5, 4 * w // 5),
        "y": h - rng.randint(0, h // 10),
        "length": rng.randint(80, 160),
        "angle": -math.pi / 2 + rng.uniform(-0.4, 0.4),
        "depth": rng.randint(3, 6),
        "branch_factor": rng.randint(2, 4),
        "color_offset": rng.random(),
        "curvature": rng.uniform(-0.25, 0.25),
    }


def create_individual(image_size: Tuple[int, int], num_fractals: int,
                      rng: random.Random) -> Individual:
    return [random_gene(image_size, rng) for _ in range(num_fractals)]


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness(individual: Individual) -> float:
    if not individual:
        return 0.0
    xs = [g["x"] for g in individual]
    lengths = [g["length"] for g in individual]

    spread = (max(xs) - min(xs)) if len(xs) > 1 else 0
    branch_score = sum(g["branch_factor"] * g["depth"] for g in individual)
    offsets = [g["color_offset"] for g in individual]
    colour_var = (max(offsets) - min(offsets)) if len(offsets) > 1 else 0
    size_var = (max(lengths) - min(lengths)) if len(lengths) > 1 else 0

    # Symmetry bonus
    if len(xs) > 1:
        mean_x = sum(xs) / len(xs)
        symmetry = 1.0 / (1.0 + sum(abs(x - mean_x) / max(1, mean_x) for x in xs) / len(xs))
    else:
        symmetry = 0.5

    return spread * 1.0 + branch_score * 2.0 + colour_var * 100.0 + size_var * 0.5 + symmetry * 50.0


# ── Diversity ─────────────────────────────────────────────────────────────────

def population_diversity(population: Population) -> float:
    """Measure genetic diversity as average pairwise distance."""
    if len(population) < 2:
        return 0.0
    sample_size = min(10, len(population))
    sample = random.sample(population, sample_size)
    total_dist, count = 0.0, 0
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dist = _individual_distance(sample[i], sample[j])
            total_dist += dist
            count += 1
    return total_dist / max(1, count)


def _individual_distance(a: Individual, b: Individual) -> float:
    if len(a) != len(b):
        return 1e6
    total = 0.0
    for g1, g2 in zip(a, b):
        for k in GENE_KEYS:
            v1, v2 = float(g1[k]), float(g2[k])
            total += (v1 - v2) ** 2
    return math.sqrt(total)


# ── Selection ─────────────────────────────────────────────────────────────────

def select_parents(population: Population, rng: random.Random) -> Tuple[Individual, Individual]:
    if len(population) < 2:
        return population[0], population[0]
    fits = np.array([fitness(ind) for ind in population], dtype=np.float64)
    fits_scaled = fits - fits.min() + 1.0
    probs = fits_scaled / fits_scaled.sum()
    i1, i2 = rng.choices(range(len(population)), weights=probs, k=2)
    while i1 == i2 and len(population) > 1:
        i2 = rng.choices(range(len(population)), weights=probs, k=1)[0]
    return population[i1], population[i2]


# ── Genetic operators ─────────────────────────────────────────────────────────

def mutate(individual: Individual, rate: float, image_size: Tuple[int, int],
           rng: random.Random) -> Individual:
    w, h = image_size
    result: Individual = []
    for gene in individual:
        ng = gene.copy()
        if rng.random() < rate:
            trait = rng.choice(list(gene.keys()))
            if trait == "x":
                ng[trait] = max(10, min(w - 10, gene[trait] + rng.randint(-15, 15)))
            elif trait == "y":
                ng[trait] = max(10, min(h - 10, gene[trait] + rng.randint(-15, 15)))
            elif trait == "depth":
                ng[trait] = max(2, min(7, gene[trait] + rng.randint(-1, 1)))
            elif trait == "branch_factor":
                ng[trait] = max(2, min(5, gene[trait] + rng.randint(-1, 1)))
            elif trait == "length":
                ng[trait] = max(30, min(220, gene[trait] + rng.randint(-20, 20)))
            elif trait == "angle":
                ng[trait] = gene[trait] + rng.uniform(-0.2, 0.2)
            elif trait == "color_offset":
                ng[trait] = max(0, min(1, gene[trait] + rng.uniform(-0.15, 0.15)))
            elif trait == "curvature":
                ng[trait] = gene[trait] + rng.uniform(-0.1, 0.1)
        result.append(ng)
    return result


def crossover(p1: Individual, p2: Individual, rng: random.Random) -> Individual:
    return [
        {k: (g1[k] if rng.random() < 0.5 else g2[k]) for k in g1}
        for g1, g2 in zip(p1, p2)
    ]


# ── GA Engine ─────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """Full evolutionary lifecycle with smooth transitions, save/load, and adaptive mutation."""

    def __init__(
        self,
        image_size: Tuple[int, int] | None = None,
        pop_size: int | None = None,
        num_fractals: int | None = None,
        mutation_rate: float | None = None,
        frames_per_gen: int | None = None,
        seed: int | None = None,
        adaptive_mutation: bool = True,
    ):
        self.image_size = image_size or IMAGE_SIZES[DEFAULTS["image_size_key"]]
        self.pop_size = max(2, pop_size or DEFAULTS["population_size"])
        self.num_fractals = max(1, num_fractals or DEFAULTS["num_fractals"])
        self.base_mutation_rate = mutation_rate if mutation_rate is not None else DEFAULTS["mutation_rate"]
        self.mutation_rate = self.base_mutation_rate
        self.frames_per_gen = max(1, frames_per_gen or DEFAULTS["frames_per_gen"])
        self.adaptive_mutation = adaptive_mutation

        self._seed = seed if seed is not None else DEFAULTS["seed"]
        self.rng = random.Random(self._seed)

        self.population: Population = []
        self.next_population: Population = []
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self._stagnation_count = 0

        self.initialize()

    def initialize(self) -> None:
        self.population = [
            create_individual(self.image_size, self.num_fractals, self.rng)
            for _ in range(self.pop_size)
        ]
        self.next_population = [self._breed_child() for _ in range(self.pop_size)]
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self._stagnation_count = 0
        self.mutation_rate = self.base_mutation_rate

        fits = [fitness(ind) for ind in self.population]
        self.best_fitness = max(fits)
        self.avg_fitness = sum(fits) / len(fits)
        self.fitness_history = [self.best_fitness]
        self.avg_fitness_history = [self.avg_fitness]
        self.diversity_history = [population_diversity(self.population)]

    def get_interpolated_population(self, easing: str = "smoothstep") -> Population:
        from interpolation import interpolate_individual
        t = self.frame_idx / max(1, self.frames_per_gen)
        return [
            interpolate_individual(ind, nxt, t, easing)
            for ind, nxt in zip(self.population, self.next_population)
        ]

    def step(self) -> bool:
        self.frame_idx += 1
        self.global_time += 0.1
        if self.frame_idx >= self.frames_per_gen:
            self._evolve()
            return True
        return False

    def _breed_child(self) -> Individual:
        p1, p2 = select_parents(self.population, self.rng)
        child = crossover(p1, p2, self.rng)
        return mutate(child, self.mutation_rate, self.image_size, self.rng)

    def _evolve(self) -> None:
        self.population = self.next_population[:]
        self.population.sort(key=fitness, reverse=True)

        fits = [fitness(ind) for ind in self.population]
        prev_best = self.best_fitness
        self.best_fitness = fits[0]
        self.avg_fitness = sum(fits) / len(fits)
        self.fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(self.avg_fitness)

        # Diversity tracking
        div = population_diversity(self.population)
        self.diversity_history.append(div)

        # Adaptive mutation
        if self.adaptive_mutation:
            improvement = self.best_fitness - prev_best
            if improvement < 1.0:
                self._stagnation_count += 1
            else:
                self._stagnation_count = max(0, self._stagnation_count - 1)
            # Increase mutation when stagnating, decrease when improving
            boost = min(0.3, self._stagnation_count * 0.02)
            self.mutation_rate = min(0.8, self.base_mutation_rate + boost)

        # Elitism
        elite_count = max(1, self.pop_size // 3)
        elite = self.population[:elite_count]

        new_next = list(elite)
        while len(new_next) < self.pop_size:
            new_next.append(self._breed_child())

        self.next_population = new_next
        self.generation += 1
        self.frame_idx = 0

    def get_best_individual(self) -> Optional[Individual]:
        if not self.population:
            return None
        return max(self.population, key=fitness)

    def get_stats(self) -> dict:
        return {
            "generation": self.generation,
            "frame": self.frame_idx,
            "frames_per_gen": self.frames_per_gen,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "pop_size": self.pop_size,
            "global_time": self.global_time,
            "mutation_rate": self.mutation_rate,
            "diversity": self.diversity_history[-1] if self.diversity_history else 0.0,
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version": "2.0",
            "image_size": list(self.image_size),
            "pop_size": self.pop_size,
            "num_fractals": self.num_fractals,
            "base_mutation_rate": self.base_mutation_rate,
            "mutation_rate": self.mutation_rate,
            "frames_per_gen": self.frames_per_gen,
            "seed": self._seed,
            "adaptive_mutation": self.adaptive_mutation,
            "generation": self.generation,
            "frame_idx": self.frame_idx,
            "global_time": self.global_time,
            "population": self.population,
            "next_population": self.next_population,
            "fitness_history": self.fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "diversity_history": self.diversity_history,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneticAlgorithm":
        ga = cls.__new__(cls)
        ga.image_size = tuple(data["image_size"])
        ga.pop_size = data["pop_size"]
        ga.num_fractals = data["num_fractals"]
        ga.base_mutation_rate = data.get("base_mutation_rate", data.get("mutation_rate", 0.2))
        ga.mutation_rate = data.get("mutation_rate", ga.base_mutation_rate)
        ga.frames_per_gen = data["frames_per_gen"]
        ga._seed = data.get("seed", 42)
        ga.rng = random.Random(ga._seed)
        ga.adaptive_mutation = data.get("adaptive_mutation", True)
        ga.generation = data["generation"]
        ga.frame_idx = data["frame_idx"]
        ga.global_time = data["global_time"]
        ga.population = data["population"]
        ga.next_population = data["next_population"]
        ga.fitness_history = data.get("fitness_history", [])
        ga.avg_fitness_history = data.get("avg_fitness_history", [])
        ga.diversity_history = data.get("diversity_history", [])
        ga.best_fitness = data.get("best_fitness", 0.0)
        ga.avg_fitness = data.get("avg_fitness", 0.0)
        ga._stagnation_count = 0
        return ga