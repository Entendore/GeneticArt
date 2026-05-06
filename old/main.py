import numpy as np
from PIL import Image, ImageDraw
import random, os

# --- Parameters ---
IMAGE_SIZE = (512, 512)
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.2
NUM_FRACTALS = 5  # Number of fractal trees per individual
OUTPUT_DIR = "fractal_genetic_art"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Harmonic color palettes ---
def harmonic_color(index, total):
    # Return an RGB value cycling through hues harmonically
    hue = index / total
    r = int(127 * (np.sin(2*np.pi*hue) + 1))
    g = int(127 * (np.sin(2*np.pi*hue + 2*np.pi/3) + 1))
    b = int(127 * (np.sin(2*np.pi*hue + 4*np.pi/3) + 1))
    return (r, g, b)

# --- Gene definition ---
# Each fractal tree: [x, y, length, angle, depth, branch_factor, color_offset]
def random_gene():
    return [
        random.randint(IMAGE_SIZE[0]//4, 3*IMAGE_SIZE[0]//4),  # x
        IMAGE_SIZE[1],                                        # y (bottom)
        random.randint(50, 150),                              # length
        random.uniform(-np.pi/2-0.3, -np.pi/2+0.3),          # angle (upwards)
        random.randint(3, 6),                                 # depth
        random.randint(2, 4),                                 # branch factor
        random.random()                                       # color offset
    ]

def create_individual():
    return [random_gene() for _ in range(NUM_FRACTALS)]

# --- Fitness function ---
# Favor longer branches and deeper trees
def fitness(individual):
    score = 0
    for gene in individual:
        score += gene[2] * gene[4]  # length * depth
    return score

# --- Genetic operations ---
def mutate(individual):
    new_individual = []
    for gene in individual:
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, len(gene)-1)
            gene[idx] = random_gene()[idx]
        new_individual.append(gene)
    return new_individual

def crossover(parent1, parent2):
    point = random.randint(0, NUM_FRACTALS-1)
    return parent1[:point] + parent2[point:]

# --- Recursive fractal rendering ---
def draw_fractal(draw, x, y, length, angle, depth, branch_factor, color_offset):
    if depth == 0 or length < 2:
        return
    x2 = x + int(np.cos(angle) * length)
    y2 = y + int(np.sin(angle) * length)
    
    color = harmonic_color(int(color_offset*1000), 1000)
    alpha = int(50 + 205 * depth / 6)
    draw.line((x, y, x2, y2), fill=color + (alpha,), width=depth)
    
    for i in range(branch_factor):
        new_angle = angle + np.random.uniform(-0.5, 0.5)
        new_length = length * np.random.uniform(0.6, 0.8)
        draw_fractal(draw, x2, y2, new_length, new_angle, depth-1, branch_factor, color_offset + i*0.05)

# --- Rendering function ---
def render(individual, filename):
    img = Image.new("RGBA", IMAGE_SIZE, (0,0,0,255))
    draw = ImageDraw.Draw(img, 'RGBA')
    for gene in individual:
        x, y, length, angle, depth, branch_factor, color_offset = gene
        draw_fractal(draw, x, y, length, angle, depth, branch_factor, color_offset)
    img.save(filename)

# --- Main Genetic Loop ---
population = [create_individual() for _ in range(POPULATION_SIZE)]

for gen in range(GENERATIONS):
    population.sort(key=fitness, reverse=True)
    best_filename = os.path.join(OUTPUT_DIR, f"gen_{gen:03d}.png")
    render(population[0], best_filename)
    print(f"Generation {gen}, Fitness: {fitness(population[0])}")
    
    # Create next generation
    new_population = population[:2]  # Elitism
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = random.choices(population[:5], k=2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    population = new_population
