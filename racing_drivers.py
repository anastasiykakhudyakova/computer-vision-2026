'''нужно сделать программу которая будет симулировать едущих машинки по гоночной трассе. управляется нейросетью, нейросеть должна оптимизироваться генетическим алгоритмом'''
import pygame
import numpy as np
import math
import random

pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing AI - Genetic Algorithm")
clock = pygame.time.Clock()
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

TRACK_COLOR = DARK_GRAY
BORDER_COLOR = WHITE
BG_COLOR = BLACK

track_points = [
    (200, 150), (800, 150), (900, 350), (800, 550),
    (200, 550), (100, 350)
]
track_width = 80

track_surface = pygame.Surface((WIDTH, HEIGHT))
track_surface.fill(BG_COLOR)
pygame.draw.polygon(track_surface, TRACK_COLOR, track_points)

for i in range(len(track_points)):
    p1 = track_points[i]
    p2 = track_points[(i+1)%len(track_points)]
    pygame.draw.line(track_surface, BORDER_COLOR, p1, p2, 6)

checkpoints = []
num_segments = 20
for i in range(len(track_points)):
    p1 = np.array(track_points[i])
    p2 = np.array(track_points[(i+1)%len(track_points)])
    for t in np.linspace(0, 1, num_segments//len(track_points) + 1):
        pt = p1 + t*(p2 - p1)
        checkpoints.append(tuple(pt.astype(int)))
checkpoints = checkpoints[:-1]

INPUT_SIZE = 6
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2

POP_SIZE = 30
SPAWN_TIMEOUT = 3.0
MUTATION_RATE = 0.3
MUTATION_SCALE = 1.0
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.7

class Car:
    def __init__(self, weights=None):
        self.start_x, self.start_y = track_points[0]
        self.x = self.start_x
        self.y = self.start_y
        self.angle = 90
        self.speed = random.uniform(10, 50)
        self.alive = True
        self.distance = 0.0
        self.last_pos = (self.x, self.y)
        self.sensor_angles = [-90, -45, 0, 45, 90]
        self.sensor_length = 200
        self.sensor_readings = [self.sensor_length] * len(self.sensor_angles)
        self.nn = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        if weights is not None:
            self.nn.set_weights(weights)
        self.fitness = 0
        self.checkpoint_idx = 0
        self.time_alive = 0
        self.stuck_counter = 0
        self.last_distance_check = 0

    def is_on_track(self, x, y):
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False
        color = track_surface.get_at((int(x), int(y)))
        return color != BG_COLOR

    def update(self, dt):
        if not self.alive:
            return
        self.time_alive += dt
        self.read_sensors()
        
        sensor_inputs = [s / self.sensor_length for s in self.sensor_readings]
        speed_input = self.speed / 200.0
        
        inputs = np.array(sensor_inputs + [speed_input])
        outputs = self.nn.forward(inputs)
        
        turn = (outputs[0] - 0.5) * 2
        accel = (outputs[1] - 0.5) * 2

        self.angle += turn * 180 * dt
        self.speed += accel * 100 * dt
        self.speed = max(0, min(self.speed, 200))
        self.speed *= 0.99

        dx = math.cos(math.radians(self.angle)) * self.speed * dt
        dy = math.sin(math.radians(self.angle)) * self.speed * dt
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        if not self.is_on_track(new_x, new_y):
            self.alive = False
            return
            
        self.x = new_x
        self.y = new_y

        dist = math.hypot(self.x - self.last_pos[0], self.y - self.last_pos[1])
        self.distance += dist
        self.last_pos = (self.x, self.y)
        
        if self.distance - self.last_distance_check < 5 and self.time_alive > 2:
            self.stuck_counter += 1
            if self.stuck_counter > 60:
                self.alive = False
                return
        else:
            self.stuck_counter = 0
            self.last_distance_check = self.distance

        for i in range(self.checkpoint_idx, len(checkpoints)):
            cp = checkpoints[i]
            if math.hypot(self.x - cp[0], self.y - cp[1]) < 40:
                self.checkpoint_idx = i + 1
                if self.checkpoint_idx >= len(checkpoints):
                    self.checkpoint_idx = 0
                break

    def read_sensors(self):
        for i, angle in enumerate(self.sensor_angles):
            rad = math.radians(self.angle + angle)
            dist = self.sensor_length
            
            for d in range(0, self.sensor_length, 5):
                x = int(self.x + math.cos(rad) * d)
                y = int(self.y + math.sin(rad) * d)
                
                if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                    dist = d
                    break
                    
                color = track_surface.get_at((x, y))
                if color == BG_COLOR:
                    dist = d
                    break
                    
            self.sensor_readings[i] = dist

    def calculate_fitness(self):
        self.fitness = self.distance + self.checkpoint_idx * 200 - self.time_alive * 0.5
        return self.fitness

    def draw(self, surf, best=False):
        if not self.alive:
            return
        color = GREEN if best else RED
        
        angle_rad = math.radians(self.angle)
        car_points = [
            (self.x + math.cos(angle_rad) * 10, self.y + math.sin(angle_rad) * 10),
            (self.x + math.cos(angle_rad + 2.5) * 7, self.y + math.sin(angle_rad + 2.5) * 7),
            (self.x + math.cos(angle_rad - 2.5) * 7, self.y + math.sin(angle_rad - 2.5) * 7)
        ]
        pygame.draw.polygon(surf, color, car_points)
        pygame.draw.circle(surf, WHITE, (int(self.x), int(self.y)), 2)
        
        if best:
            for i, angle in enumerate(self.sensor_angles):
                rad = math.radians(self.angle + angle)
                end_x = self.x + math.cos(rad) * self.sensor_readings[i]
                end_y = self.y + math.sin(rad) * self.sensor_readings[i]
                pygame.draw.line(surf, YELLOW, (self.x, self.y), (end_x, end_y), 1)

class NeuralNet:
    def __init__(self, in_size, hid_size, out_size):
        self.w1 = np.random.randn(in_size, hid_size) * 0.5
        self.b1 = np.random.randn(1, hid_size) * 0.1
        self.w2 = np.random.randn(hid_size, out_size) * 0.5
        self.b2 = np.random.randn(1, out_size) * 0.1

    def forward(self, x):
        x = np.array(x, ndmin=2)
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))
        return self.a2[0]

    def get_weights(self):
        return np.concatenate([
            self.w1.flatten(), self.b1.flatten(),
            self.w2.flatten(), self.b2.flatten()
        ])

    def set_weights(self, flat_weights):
        idx = 0
        size_w1 = self.w1.size
        self.w1 = flat_weights[idx:idx+size_w1].reshape(self.w1.shape)
        idx += size_w1
        size_b1 = self.b1.size
        self.b1 = flat_weights[idx:idx+size_b1].reshape(self.b1.shape)
        idx += size_b1
        size_w2 = self.w2.size
        self.w2 = flat_weights[idx:idx+size_w2].reshape(self.w2.shape)
        idx += size_w2
        size_b2 = self.b2.size
        self.b2 = flat_weights[idx:idx+size_b2].reshape(self.b2.shape)

    @staticmethod
    def crossover(parent1_weights, parent2_weights):
        child_weights = np.zeros_like(parent1_weights)
        for i in range(len(parent1_weights)):
            if random.random() < CROSSOVER_RATE:
                child_weights[i] = parent1_weights[i] if random.random() < 0.5 else parent2_weights[i]
            else:
                child_weights[i] = random.choice([parent1_weights[i], parent2_weights[i]])
        return child_weights

    @staticmethod
    def mutate(weights):
        mutation_mask = np.random.rand(len(weights)) < MUTATION_RATE
        noise = np.random.randn(len(weights)) * MUTATION_SCALE
        weights += mutation_mask * noise
        return weights

def generate_population():
    pop = []
    for _ in range(POP_SIZE):
        car = Car()
        pop.append(car)
    return pop

def tournament_selection(population, fitnesses):
    best = None
    best_fit = -np.inf
    for _ in range(TOURNAMENT_SIZE):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best = population[idx]
            best_fit = fitnesses[idx]
    return best

def evolve(population):
    fitnesses = [car.calculate_fitness() for car in population]
    new_population = []
    
    sorted_indices = np.argsort(fitnesses)[::-1]
    for i in range(min(5, len(sorted_indices))):
        best_idx = sorted_indices[i]
        best_car = population[best_idx]
        best_weights = best_car.nn.get_weights()
        new_population.append(Car(weights=best_weights.copy()))

    while len(new_population) < POP_SIZE:
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child_weights = NeuralNet.crossover(parent1.nn.get_weights(), parent2.nn.get_weights())
        child_weights = NeuralNet.mutate(child_weights)
        new_population.append(Car(weights=child_weights))
    return new_population, max(fitnesses)

def spawn_car(population, best_weights=None):
    if best_weights is not None:
        weights = best_weights.copy()
        weights = NeuralNet.mutate(weights)
        car = Car(weights=weights)
    else:
        car = Car()
    population.append(car)
    return car

def main():
    population = generate_population()
    generation = 0
    running = True
    best_car = None
    best_fitness = -np.inf
    spawn_timer = 0
    generation_timer = 0
    max_generation_time = 15.0

    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        spawn_timer += dt
        generation_timer += dt
        
        if spawn_timer >= SPAWN_TIMEOUT:
            spawn_timer = 0
            if best_car is not None:
                spawn_car(population, best_car.nn.get_weights())
            else:
                spawn_car(population)

        alive_count = 0
        for car in population:
            if car.alive:
                car.update(dt)
                alive_count += 1
        
        if alive_count == 0 or generation_timer >= max_generation_time:
            fitnesses = [car.calculate_fitness() for car in population]
            gen_best_fitness = max(fitnesses)
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_idx = np.argmax(fitnesses)
                best_car = population[best_idx]
            
            alive_cars = [car for car in population if car.alive]
            dead_cars = [car for car in population if not car.alive]
            
            population = alive_cars[:min(10, len(alive_cars))]
            
            if len(population) < POP_SIZE:
                sorted_dead = sorted(dead_cars, key=lambda c: c.calculate_fitness(), reverse=True)
                for car in sorted_dead[:POP_SIZE - len(population)]:
                    population.append(car)
            
            while len(population) < POP_SIZE:
                if best_car is not None and random.random() < 0.7:
                    weights = best_car.nn.get_weights()
                    weights = NeuralNet.mutate(weights)
                    population.append(Car(weights=weights))
                else:
                    population.append(Car())
            
            for car in population:
                car.x = car.start_x
                car.y = car.start_y
                car.angle = 90
                car.speed = random.uniform(10, 50)
                car.alive = True
                car.distance = 0
                car.last_pos = (car.x, car.y)
                car.checkpoint_idx = 0
                car.time_alive = 0
                car.fitness = 0
                car.stuck_counter = 0
                car.last_distance_check = 0
            
            generation += 1
            spawn_timer = 0
            generation_timer = 0

        screen.fill(BLACK)
        screen.blit(track_surface, (0, 0))
        
        for cp in checkpoints:
            pygame.draw.circle(screen, YELLOW, cp, 3)

        if best_car is not None and best_car.alive:
            best_car.draw(screen, best=True)
        
        for car in population:
            if car.alive and car != best_car:
                car.draw(screen)

        font = pygame.font.Font(None, 24)
        gen_text = font.render(f"Generation: {generation}", True, WHITE)
        alive_text = font.render(f"Alive: {alive_count}/{len(population)}", True, WHITE)
        fit_text = font.render(f"Best fitness: {best_fitness:.1f}", True, WHITE)
        spawn_text = font.render(f"Spawn in: {max(0, SPAWN_TIMEOUT - spawn_timer):.1f}s", True, WHITE)
        
        screen.blit(gen_text, (10, 10))
        screen.blit(alive_text, (10, 30))
        screen.blit(fit_text, (10, 50))
        screen.blit(spawn_text, (10, 70))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
