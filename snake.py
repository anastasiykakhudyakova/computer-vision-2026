'''научить агента играть в змейку с Q обучением'''
import pygame
import numpy as np
import random
from collections import deque
import pickle
import os

pygame.init()
WINDOW_SIZE = 600
CELL_SIZE = 30
GRID_SIZE = WINDOW_SIZE // CELL_SIZE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 60))
pygame.display.set_caption("Snake Q-Learning")
clock = pygame.time.Clock()

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 1000

DIRECTIONS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0)
}

class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = 0
        self.food = self.generate_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        return self.get_state()
    
    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food
    
    def get_state(self):
        head = self.snake[0]
        
        danger_straight = self.check_danger(self.direction)
        danger_right = self.check_danger((self.direction + 1) % 4)
        danger_left = self.check_danger((self.direction - 1) % 4)
        
        dir_up = self.direction == 0
        dir_right = self.direction == 1
        dir_down = self.direction == 2
        dir_left = self.direction == 3
        
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        state = [
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_left, food_right, food_up, food_down
        ]
        
        return tuple(int(s) for s in state)
    
    def check_danger(self, direction):
        head = self.snake[0]
        next_pos = (head[0] + DIRECTIONS[direction][0], head[1] + DIRECTIONS[direction][1])
        
        if (next_pos[0] < 0 or next_pos[0] >= GRID_SIZE or 
            next_pos[1] < 0 or next_pos[1] >= GRID_SIZE):
            return True
        
        if next_pos in self.snake[1:]:
            return True
        
        return False
    
    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True
        
        self.steps += 1
        
        if action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
        
        head = self.snake[0]
        new_head = (head[0] + DIRECTIONS[self.direction][0], 
                   head[1] + DIRECTIONS[self.direction][1])
        
        reward = 0
        
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or 
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or 
            new_head in self.snake):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self.generate_food()
            self.steps = 0
        else:
            self.snake.pop()
            reward = -0.1
        
        if self.steps > 100:
            self.game_over = True
            reward = -10
        
        return self.get_state(), reward, self.game_over
    
    def render(self, screen):
        screen.fill(BLACK)
        
        for i, segment in enumerate(self.snake):
            color = DARK_GREEN if i == 0 else GREEN
            pygame.draw.rect(screen, color, 
                           (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + 60, 
                            CELL_SIZE - 2, CELL_SIZE - 2))
        
        pygame.draw.rect(screen, RED,
                        (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE + 60,
                         CELL_SIZE - 2, CELL_SIZE - 2))
        
        font = pygame.font.Font(None, 24)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        
        self.train_step = 0
    
    def build_model(self):
        model = []
        
        w1 = np.random.randn(self.state_size, 128) * 0.1
        b1 = np.zeros(128)
        w2 = np.random.randn(128, 128) * 0.1
        b2 = np.zeros(128)
        w3 = np.random.randn(128, self.action_size) * 0.1
        b3 = np.zeros(self.action_size)
        
        return [w1, b1, w2, b2, w3, b3]
    
    def forward(self, state, model):
        x = np.array(state).reshape(1, -1)
        
        z1 = np.dot(x, model[0]) + model[1]
        a1 = np.maximum(0, z1)
        
        z2 = np.dot(a1, model[2]) + model[3]
        a2 = np.maximum(0, z2)
        
        z3 = np.dot(a2, model[4]) + model[5]
        
        return z3[0]
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.forward(state, self.model)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        targets = np.zeros((BATCH_SIZE, self.action_size))
        
        for i in range(BATCH_SIZE):
            targets[i] = self.forward(states[i], self.model)
            
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                next_q = self.forward(next_states[i], self.target_model)
                targets[i][actions[i]] = rewards[i] + DISCOUNT_FACTOR * np.max(next_q)
        
        for epoch in range(1):
            for i in range(BATCH_SIZE):
                x = states[i].reshape(1, -1)
                
                z1 = np.dot(x, self.model[0]) + self.model[1]
                a1 = np.maximum(0, z1)
                
                z2 = np.dot(a1, self.model[2]) + self.model[3]
                a2 = np.maximum(0, z2)
                
                output = np.dot(a2, self.model[4]) + self.model[5]
                
                error = output - targets[i].reshape(1, -1)
                
                delta3 = error
                delta2 = np.dot(delta3, self.model[4].T) * (z2 > 0)
                delta1 = np.dot(delta2, self.model[2].T) * (z1 > 0)
                
                self.model[4] -= LEARNING_RATE * np.dot(a2.T, delta3)
                self.model[5] -= LEARNING_RATE * delta3[0]
                self.model[2] -= LEARNING_RATE * np.dot(a1.T, delta2)
                self.model[3] -= LEARNING_RATE * delta2[0]
                self.model[0] -= LEARNING_RATE * np.dot(x.T, delta1)
                self.model[1] -= LEARNING_RATE * delta1[0]
        
        self.train_step += 1
        
        if self.train_step % TARGET_UPDATE_FREQ == 0:
            self.update_target_model()
        
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def update_target_model(self):
        for i in range(len(self.model)):
            self.target_model[i] = self.model[i].copy()
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'epsilon': self.epsilon
            }, f)
    
    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.target_model = [m.copy() for m in self.model]
                self.epsilon = data['epsilon']

def main():
    game = SnakeGame()
    agent = DQNAgent(11, 3)
    
    try:
        agent.load("snake_model.pkl")
        print("Model loaded successfully")
    except:
        print("Starting new training")
    
    episodes = 0
    total_score = 0
    best_score = 0
    running = True
    paused = False
    speed = 10
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(1000, speed + 10)
                elif event.key == pygame.K_DOWN:
                    speed = max(10, speed - 10)
                elif event.key == pygame.K_s:
                    agent.save("snake_model.pkl")
                    print("Model saved")
        
        if not paused:
            state = game.get_state()
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            if done:
                total_score += game.score
                episodes += 1
                
                if game.score > best_score:
                    best_score = game.score
                
                if episodes % 100 == 0:
                    avg_score = total_score / 100
                    print(f"Episode: {episodes}, Avg Score: {avg_score:.2f}, "
                          f"Best: {best_score}, Epsilon: {agent.epsilon:.3f}")
                    total_score = 0
                    
                    if avg_score > 20:
                        agent.save("snake_model.pkl")
                        print(f"Model saved with avg score: {avg_score:.2f}")
                
                game.reset()
        
        game.render(screen)
        
        font = pygame.font.Font(None, 24)
        episode_text = font.render(f"Episode: {episodes}", True, WHITE)
        best_text = font.render(f"Best: {best_score}", True, WHITE)
        epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, WHITE)
        speed_text = font.render(f"Speed: {speed}", True, WHITE)
        status_text = font.render("PAUSED" if paused else "RUNNING", True, 
                                 RED if paused else GREEN)
        
        info_y = WINDOW_SIZE + 10
        screen.blit(episode_text, (10, info_y))
        screen.blit(best_text, (150, info_y))
        screen.blit(epsilon_text, (300, info_y))
        screen.blit(speed_text, (450, info_y))
        screen.blit(status_text, (10, info_y + 20))
        
        pygame.display.flip()
        clock.tick(speed)
    
    agent.save("snake_model.pkl")
    pygame.quit()

if __name__ == "__main__":
    main()
