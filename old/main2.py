import numpy as np
from PIL import Image, ImageDraw
import random, os

# --- Parameters ---
IMAGE_SIZE = (512, 512)
POPULATION_SIZE = 5
GENERATIONS = 50
FRAMES_PER_GENERATION = 20
MUTATION_RATE = 0.2
NUM_FRACTALS = 3
OUTPUT_DIR = "fractal_animation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Harmonic color function ---
def harmonic_color(index, total):
    hue = index / total
    r = int(127 * (np.sin(2*np.pi*hue) + 1))
    g = int(127 * (np.sin(2*np.pi*hue + 2*np.pi/3) + 1))
    b = int(127 * (np.sin(2*np.pi*hue + 4*np.pi/3) + 1))
    return (r, g, b)

# --- Gene Definition ---
# [x, y, length, angle, depth, branch_factor, color_offset]
def random_gene():
    return [
        random.randint(IMAGE_SIZE[0]//4, 3*IMAGE_SIZE[0]//4),
        IMAGE_SIZE[1],
        random.randint(50, 150),
        random.uniform(-np.pi/2-0.3, -np.pi/2+0.3),
        random.randint(3, 6),
        random.randint(2, 4),
        random.random()
    ]

def create_individual():
    return [random_gene() for _ in range(NUM_FRACTALS)]

# --- Fitness function ---
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

# --- Fractal drawing ---
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

# --- Rendering ---
def render(individual, filename):
    img = Image.new("RGBA", IMAGE_SIZE, (0,0,0,255))
    draw = ImageDraw.Draw(img, 'RGBA')
    for gene in individual:
        x, y, length, angle, depth, branch_factor, color_offset = gene
        draw_fractal(draw, x, y, length, angle, depth, branch_factor, color_offset)
    img.save(filename)

# --- Interpolation for smooth animation ---
def interpolate_gene(g1, g2, t):
    return [g1[i]*(1-t) + g2[i]*t if isinstance(g1[i], float) else int(g1[i]*(1-t) + g2[i]*t) for i in range(len(g1))]

def interpolate_individual(ind1, ind2, t):
    return [interpolate_gene(ind1[i], ind2[i], t) for i in range(NUM_FRACTALS)]

# --- Main Evolutionary Animation Loop ---
population = [create_individual() for _ in range(POPULATION_SIZE)]
frame_count = 0

for gen in range(GENERATIONS):
    population.sort(key=fitness, reverse=True)
    # Select top two as parents for next generation
    parent1 = population[0]
    parent2 = population[1]
    
    # Generate next generation child
    child = mutate(crossover(parent1, parent2))
    
    # Smooth interpolation frames
    for f in range(FRAMES_PER_GENERATION):
        t = f / FRAMES_PER_GENERATION
        frame_individual = interpolate_individual(parent1, child, t)
        filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:05d}.png")
        render(frame_individual, filename)
        frame_count += 1
    
    # Replace worst individual with new child
    population[-1] = child
    print(f"Generation {gen} completed, frames generated: {frame_count}")
