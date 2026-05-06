import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from noise import pnoise1

# --- Parameters ---
IMAGE_SIZE = (600, 600)
NUM_LAYERS = 3
NUM_FRACTALS = 3
MUTATION_RATE = 0.2
PERLIN_SCALE = 0.08
FRAMES_PER_GEN = 20
ROTATION_SPEED = 0.002  # Global swirling speed

# --- Harmonic color function ---
def harmonic_color(index, total, t_shift):
    hue = (index / total + t_shift) % 1.0
    r = (np.sin(2*np.pi*hue)+1)/2
    g = (np.sin(2*np.pi*hue + 2*np.pi/3)+1)/2
    b = (np.sin(2*np.pi*hue + 4*np.pi/3)+1)/2
    return np.array([r, g, b])

# --- Gene Definition ---
def random_gene():
    return [
        random.randint(IMAGE_SIZE[0]//4, 3*IMAGE_SIZE[0]//4),  # x
        IMAGE_SIZE[1],                                        # y
        random.randint(80, 160),                              # length
        random.uniform(-np.pi/2-0.3, -np.pi/2+0.3),          # angle
        random.randint(3, 6),                                 # depth
        random.randint(2, 4),                                 # branch_factor
        random.random()                                       # color_offset
    ]

def create_layer():
    return [random_gene() for _ in range(NUM_FRACTALS)]

def mutate_layer(layer):
    new_layer = []
    for gene in layer:
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, len(gene)-1)
            gene[idx] = random_gene()[idx]
        new_layer.append(gene)
    return new_layer

def crossover_layer(parent1, parent2):
    point = random.randint(0, NUM_FRACTALS-1)
    return parent1[:point] + parent2[point:]

# --- Fractal drawing with Perlin noise and swirling ---
def draw_fractal(ax, x, y, length, angle, depth, branch_factor, color_offset, t, alpha=0.5, swirl=0):
    if depth == 0 or length < 2:
        return
    angle += pnoise1(t + x*PERLIN_SCALE, repeat=1024) * 0.5 + swirl
    x2 = x + np.cos(angle) * length
    y2 = y + np.sin(angle) * length

    color = harmonic_color(int(color_offset*1000), 1000, t*0.01)
    ax.plot([x, x2], [y, y2], color=color, alpha=alpha, linewidth=depth)

    for i in range(branch_factor):
        new_angle = angle + np.random.uniform(-0.5, 0.5)
        new_length = length * np.random.uniform(0.6, 0.8)
        draw_fractal(ax, x2, y2, new_length, new_angle, depth-1, branch_factor, color_offset + i*0.05, t, alpha, swirl)

# --- Interpolation ---
def interpolate_gene(g1, g2, t):
    return [g1[i]*(1-t) + g2[i]*t if isinstance(g1[i], float) else int(g1[i]*(1-t) + g2[i]*t) for i in range(len(g1))]

def interpolate_layer(layer1, layer2, t):
    return [interpolate_gene(layer1[i], layer2[i], t) for i in range(NUM_FRACTALS)]

# --- Initialize layers ---
layers = [create_layer() for _ in range(NUM_LAYERS)]
next_layers = [mutate_layer(crossover_layer(layers[i], layers[i])) for i in range(NUM_LAYERS)]
frame_idx = 0
global_time = 0

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, IMAGE_SIZE[0])
ax.set_ylim(0, IMAGE_SIZE[1])
ax.axis('off')

# --- Gradient background helper ---
def draw_gradient(ax, color_top, color_bottom):
    gradient = np.linspace(0,1,IMAGE_SIZE[1])
    gradient = np.outer(np.ones(IMAGE_SIZE[0]), gradient)
    gradient_img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
    for i in range(3):
        gradient_img[:,:,i] = color_bottom[i] * (1-gradient) + color_top[i] * gradient
    ax.imshow(gradient_img, origin='lower', extent=[0, IMAGE_SIZE[0], 0, IMAGE_SIZE[1]])

# --- Animation update ---
def update(frame):
    global layers, next_layers, frame_idx, global_time
    ax.clear()
    ax.set_xlim(0, IMAGE_SIZE[0])
    ax.set_ylim(0, IMAGE_SIZE[1])
    ax.axis('off')

    t = frame_idx / FRAMES_PER_GEN

    # Compute average fractal color for gradient
    avg_color = np.zeros(3)
    for layer in layers:
        for gene in layer:
            avg_color += harmonic_color(int(gene[6]*1000), 1000, global_time*0.01)
    avg_color /= (NUM_LAYERS * NUM_FRACTALS)
    draw_gradient(ax, color_top=avg_color*0.9, color_bottom=avg_color*0.2)

    # Draw fractals per layer with swirling
    for i in range(NUM_LAYERS):
        alpha = 0.25 + 0.25*i
        swirl_angle = global_time * ROTATION_SPEED * (i+1)
        layer_interp = interpolate_layer(layers[i], next_layers[i], t)
        for gene in layer_interp:
            draw_fractal(ax, *gene, t=global_time, alpha=alpha, swirl=swirl_angle)

    frame_idx += 1
    global_time += 0.1

    if frame_idx >= FRAMES_PER_GEN:
        for i in range(NUM_LAYERS):
            parent1, parent2 = layers[i], next_layers[i]
            child = mutate_layer(crossover_layer(parent1, parent2))
            layers[i] = next_layers[i]
            next_layers[i] = child
        frame_idx = 0

# --- Run animation ---
ani = FuncAnimation(fig, update, frames=2000, interval=50)
plt.show()
