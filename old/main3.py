import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# --- Parameters ---
IMAGE_SIZE = (512, 512)
POPULATION_SIZE = 5
NUM_FRACTALS = 3
MUTATION_RATE = 0.2
FRAMES_PER_GEN = 20

# --- Harmonic color function ---
def harmonic_color(index, total):
    hue = index / total
    r = (np.sin(2*np.pi*hue)+1)/2
    g = (np.sin(2*np.pi*hue + 2*np.pi/3)+1)/2
    b = (np.sin(2*np.pi*hue + 4*np.pi/3)+1)/2
    return (r, g, b)

# --- Gene Definition ---
def random_gene():
    return [
        random.randint(IMAGE_SIZE[0]//4, 3*IMAGE_SIZE[0]//4),  # x
        IMAGE_SIZE[1],                                        # y
        random.randint(50, 150),                              # length
        random.uniform(-np.pi/2-0.3, -np.pi/2+0.3),          # angle
        random.randint(3, 6),                                 # depth
        random.randint(2, 4),                                 # branch_factor
        random.random()                                       # color_offset
    ]

def create_individual():
    return [random_gene() for _ in range(NUM_FRACTALS)]

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
def draw_fractal(ax, x, y, length, angle, depth, branch_factor, color_offset):
    if depth == 0 or length < 2:
        return
    x2 = x + np.cos(angle) * length
    y2 = y + np.sin(angle) * length
    
    color = harmonic_color(int(color_offset*1000), 1000)
    ax.plot([x, x2], [y, y2], color=color, alpha=0.5, linewidth=depth)
    
    for i in range(branch_factor):
        new_angle = angle + np.random.uniform(-0.5, 0.5)
        new_length = length * np.random.uniform(0.6, 0.8)
        draw_fractal(ax, x2, y2, new_length, new_angle, depth-1, branch_factor, color_offset + i*0.05)

# --- Interpolation ---
def interpolate_gene(g1, g2, t):
    return [g1[i]*(1-t) + g2[i]*t if isinstance(g1[i], float) else int(g1[i]*(1-t) + g2[i]*t) for i in range(len(g1))]

def interpolate_individual(ind1, ind2, t):
    return [interpolate_gene(ind1[i], ind2[i], t) for i in range(NUM_FRACTALS)]

# --- Animation setup ---
population = [create_individual() for _ in range(POPULATION_SIZE)]
parent1, parent2 = population[0], population[1]
child = mutate(crossover(parent1, parent2))
frame_idx = 0

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, IMAGE_SIZE[0])
ax.set_ylim(0, IMAGE_SIZE[1])
ax.set_facecolor('black')
ax.axis('off')

def update(frame):
    global parent1, parent2, child, frame_idx, population
    
    ax.clear()
    ax.set_xlim(0, IMAGE_SIZE[0])
    ax.set_ylim(0, IMAGE_SIZE[1])
    ax.set_facecolor('black')
    ax.axis('off')
    
    t = frame_idx / FRAMES_PER_GEN
    frame_ind = interpolate_individual(parent1, child, t)
    for gene in frame_ind:
        draw_fractal(ax, *gene)
    
    frame_idx += 1
    if frame_idx >= FRAMES_PER_GEN:
        # Evolve to next generation
        population.sort(key=lambda ind: sum(g[2]*g[4] for g in ind), reverse=True)
        parent1 = population[0]
        parent2 = population[1]
        child = mutate(crossover(parent1, parent2))
        population[-1] = child
        frame_idx = 0

ani = FuncAnimation(fig, update, frames=1000, interval=50)
plt.show()
