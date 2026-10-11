import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from noise import pnoise1

# --- Parameters ---
IMAGE_SIZE = (600, 600)
POPULATION_SIZE = 5
NUM_LAYERS = 3
FRAMES_PER_GEN = 20
PERLIN_SCALE = 0.08
ROTATION_SPEED = 0.02
MUTATION_RATE = 0.2

# --- Harmonic color ---
def harmonic_color(index, total, t_shift):
    hue = (index / total + t_shift) % 1.0
    r = (np.sin(2*np.pi*hue)+1)/2
    g = (np.sin(2*np.pi*hue + 2*np.pi/3)+1)/2
    b = (np.sin(2*np.pi*hue + 4*np.pi/3)+1)/2
    return np.clip(np.array([r, g, b]), 0, 1)

# --- Gene Definition ---
def random_gene():
    return {
        'x': random.randint(IMAGE_SIZE[0]//4, 3*IMAGE_SIZE[0]//4),
        'y': 0,
        'length': random.randint(80, 160),
        'angle': np.pi/2 + random.uniform(-0.3, 0.3),
        'depth': random.randint(3, 6),
        'branch_factor': random.randint(2, 4),
        'color_offset': random.random(),
        'curvature': random.uniform(-0.2,0.2)
    }

def create_individual():
    return [random_gene() for _ in range(NUM_LAYERS)]

# --- Genetic operations ---
def mutate(individual):
    new_ind = []
    for gene in individual:
        new_gene = gene.copy()
        if random.random() < MUTATION_RATE:
            trait = random.choice(list(gene.keys()))
            if trait in ['x','y','depth','branch_factor']:
                new_gene[trait] = gene[trait] + random.randint(-10,10)
            else:
                new_gene[trait] = gene[trait] + random.uniform(-0.1,0.1)
        new_ind.append(new_gene)
    return new_ind

def crossover(parent1, parent2):
    child = []
    for g1, g2 in zip(parent1, parent2):
        new_gene = {}
        for key in g1:
            new_gene[key] = g1[key] if random.random() < 0.5 else g2[key]
        child.append(new_gene)
    return child

# --- Fitness function ---
def fitness(individual):
    spread = 0
    branch_score = 0
    color_var = 0
    xs = []
    for gene in individual:
        xs.append(gene['x'])
        branch_score += gene['branch_factor']*gene['depth']
        color_var += gene['color_offset']
    spread = max(xs) - min(xs)
    return spread + branch_score + color_var*50

# --- Selection ---
def select_parents(pop):
    fitnesses = np.array([fitness(ind) for ind in pop])
    total_fit = fitnesses.sum()
    if total_fit == 0:
        probabilities = np.ones(len(pop))/len(pop)
    else:
        probabilities = fitnesses / total_fit
    idx1, idx2 = np.random.choice(len(pop), size=2, replace=False, p=probabilities)
    return pop[idx1], pop[idx2]

# --- Fractal drawing with connected branches ---
def draw_fractal(ax, gene, t, alpha=0.5, swirl=0):
    alpha = np.clip(alpha, 0, 1)

    def recursive_draw(x, y, length, angle, depth, branch_factor, color_offset, curvature):
        if depth == 0 or length < 2:
            return

        # Compute endpoint
        angle_mod = angle + pnoise1(t + x*PERLIN_SCALE, repeat=1024)*0.5 + curvature
        x2 = x + np.cos(angle_mod)*length
        y2 = y + np.sin(angle_mod)*length

        # Apply swirling rotation for visualization
        cx, cy = IMAGE_SIZE[0]/2, 0
        dx1, dy1 = x - cx, y - cy
        dx2, dy2 = x2 - cx, y2 - cy
        cos_s, sin_s = np.cos(swirl), np.sin(swirl)
        x_rot = cx + dx1 * cos_s - dy1 * sin_s
        y_rot = cy + dx1 * sin_s + dy1 * cos_s
        x2_rot = cx + dx2 * cos_s - dy2 * sin_s
        y2_rot = cy + dx2 * sin_s + dy2 * cos_s

        # Draw the branch
        color = harmonic_color(int(color_offset*1000), 1000, t*0.01)
        ax.plot([x_rot, x2_rot], [y_rot, y2_rot], color=color, alpha=alpha, linewidth=depth)

        # Recurse from the endpoint (absolute coordinates)
        for i in range(branch_factor):
            new_angle = angle_mod + random.uniform(-0.5,0.5)
            new_length = length * random.uniform(0.6,0.8)
            recursive_draw(x2, y2, new_length, new_angle, depth-1, branch_factor, color_offset + i*0.05, curvature)

    recursive_draw(gene['x'], gene['y'], gene['length'], gene['angle'], gene['depth'],
                   gene['branch_factor'], gene['color_offset'], gene['curvature'])

# --- Interpolation ---
def interpolate_gene(g1, g2, t):
    new_gene = {}
    for key in g1:
        if isinstance(g1[key], float):
            new_gene[key] = g1[key]*(1-t)+g2[key]*t
        else:
            new_gene[key] = int(g1[key]*(1-t)+g2[key]*t)
    return new_gene

def interpolate_individual(ind1, ind2, t):
    return [interpolate_gene(g1,g2,t) for g1,g2 in zip(ind1,ind2)]

# --- Initialize population ---
population = [create_individual() for _ in range(POPULATION_SIZE)]
next_population = [mutate(crossover(*select_parents(population))) for _ in range(POPULATION_SIZE)]

frame_idx = 0
global_time = 0

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, IMAGE_SIZE[0])
ax.set_ylim(0, IMAGE_SIZE[1])
ax.axis('off')

# --- Gradient background ---
def draw_gradient(ax,color_top,color_bottom):
    gradient = np.linspace(0,1,IMAGE_SIZE[1])
    gradient = np.outer(np.ones(IMAGE_SIZE[0]), gradient)
    gradient_img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0],3))
    for i in range(3):
        gradient_img[:,:,i] = np.clip(color_bottom[i]*(1-gradient)+color_top[i]*gradient,0,1)
    ax.imshow(gradient_img, origin='lower', extent=[0,IMAGE_SIZE[0],0,IMAGE_SIZE[1]])

# --- Animation update ---
def update(frame):
    global population, next_population, frame_idx, global_time
    ax.clear()
    ax.set_xlim(0,IMAGE_SIZE[0])
    ax.set_ylim(0,IMAGE_SIZE[1])
    ax.axis('off')

    t = frame_idx / FRAMES_PER_GEN

    # Background based on average color
    avg_color = np.zeros(3)
    for ind in population:
        for gene in ind:
            avg_color += harmonic_color(int(gene['color_offset']*1000),1000,global_time*0.01)
    avg_color /= (POPULATION_SIZE*NUM_LAYERS)
    draw_gradient(ax, avg_color*0.9, avg_color*0.2)

    # Draw population
    for i, (ind, next_ind) in enumerate(zip(population, next_population)):
        swirl_angle = global_time*ROTATION_SPEED*(i+1)
        alpha = 0.25 + 0.75 * i/(POPULATION_SIZE-1)  # Always between 0.25-1.0
        interp_ind = interpolate_individual(ind, next_ind, t)
        for gene in interp_ind:
            draw_fractal(ax,gene,t,alpha,swirl_angle)

    frame_idx += 1
    global_time += 0.1

    # Evolve next generation
    if frame_idx >= FRAMES_PER_GEN:
        for i in range(POPULATION_SIZE):
            parent1, parent2 = select_parents(population)
            next_population[i] = mutate(crossover(parent1,parent2))
            population[i] = next_population[i]
        frame_idx = 0

# --- Run animation ---
ani = FuncAnimation(fig, update, frames=2000, interval=50)
plt.show()
