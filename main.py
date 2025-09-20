# AI Snake Game using Reinforcement Learning (Deep Q-Learning)
# Complete implementation with training and testing capabilities

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 640
BLOCK_SIZE = 20
GAME_WIDTH = WINDOW_WIDTH // BLOCK_SIZE
GAME_HEIGHT = WINDOW_HEIGHT // BLOCK_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class SnakeGame:
    def __init__(self, w=GAME_WIDTH, h=GAME_HEIGHT):
        self.w = w
        self.h = h
        self.reset()
        
        # Pygame display
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('AI Snake Game')
        self.clock = pygame.time.Clock()
    
    def reset(self):
        # Initialize game state
        self.direction = RIGHT
        self.head = [self.w//2, self.h//2]
        self.snake = [self.head.copy()]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()
        
    def _place_food(self):
        while True:
            x = random.randint(0, self.w-1)
            y = random.randint(0, self.h-1)
            self.food = [x, y]
            if self.food not in self.snake:
                break
    
    def play_step(self, action):
        self.frame_iteration += 1
        
        # Collect user input (for manual play)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Move snake based on action
        self._move(action)
        self.snake.insert(0, self.head.copy())
        
        # Check if game over
        reward = 0
        game_over = False
        
        # Check collisions or timeout
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -1  # Small negative reward for each step
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(10)  # Slower speed for better visibility
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        
        # Check boundary collision
        if pt[0] >= self.w or pt[0] < 0 or pt[1] >= self.h or pt[1] < 0:
            return True
        
        # Check self collision
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, 
                           pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw food
        pygame.draw.rect(self.display, RED, 
                       pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Display score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()
    
    def _move(self, action):
        # Actions: [straight, right turn, left turn]
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change (straight)
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn
        
        self.direction = new_dir
        
        x, y = self.head
        if self.direction == RIGHT:
            x += 1
        elif self.direction == LEFT:
            x -= 1
        elif self.direction == DOWN:
            y += 1
        elif self.direction == UP:
            y -= 1
        
        self.head = [x, y]

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DQNAgent:
    def __init__(self, state_size=11, action_size=3, lr=0.001):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100_000)  # replay memory
        self.model = DQN(state_size, 256, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        
        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food[0] < game.head[0],  # food left
            game.food[0] > game.head[0],  # food right
            game.food[1] < game.head[1],  # food up
            game.food[1] > game.head[1]   # food down
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])
    
    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = (dones, )
        
        # Current Q values
        pred = self.model(states)
        
        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
    
    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent()
    game = SnakeGame()
    
    print("Starting AI Snake Game Training...")
    print("The AI will learn to play Snake using Deep Q-Learning")
    print("Initial games will show poor performance as the AI explores randomly")
    print("Performance should improve significantly after 100+ games\n")
    
    while True:
        # Get current state
        state_old = agent.get_state(game)
        
        # Get move
        final_move = agent.get_action(state_old)
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Train long memory (experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                # Save model
                torch.save(agent.model.state_dict(), 'model.pth')
            
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Plot every 50 games
            if agent.n_games % 50 == 0:
                plot(plot_scores, plot_mean_scores)
                
            # Stop after good performance or many games
            if agent.n_games >= 500 or (mean_score > 15 and agent.n_games > 100):
                print(f"\nTraining completed after {agent.n_games} games!")
                print(f"Best score achieved: {record}")
                print(f"Average score: {mean_score:.2f}")
                break

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.7)
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(0.1)

def play_trained_model():
    """Play the game with a trained model"""
    agent = DQNAgent()
    
    # Load trained model
    if os.path.exists('model.pth'):
        agent.model.load_state_dict(torch.load('model.pth'))
        agent.model.eval()
        print("Loaded trained model!")
    else:
        print("No trained model found. Please train first.")
        return
    
    game = SnakeGame()
    agent.epsilon = 0  # No random moves
    
    print("Playing with trained AI agent...")
    print("Close the game window to stop.")
    
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        
        if done:
            print(f"Game Over! Final Score: {score}")
            game.reset()

def manual_play():
    """Allow manual play for testing the game environment"""
    game = SnakeGame()
    
    print("Manual Play Mode")
    print("Use WASD or Arrow Keys to control the snake")
    print("Close window to quit")
    
    while True:
        # Handle manual input
        keys = pygame.key.get_pressed()
        action = [1, 0, 0]  # Default: straight
        
        # Convert key presses to actions
        current_dir = game.direction
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if current_dir == UP:
                action = [0, 0, 1]  # left turn
            elif current_dir == DOWN:
                action = [0, 1, 0]  # right turn
            elif current_dir == LEFT:
                action = [1, 0, 0]  # straight
            elif current_dir == RIGHT:
                action = [1, 0, 0]  # straight (can't reverse)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if current_dir == UP:
                action = [0, 1, 0]  # right turn
            elif current_dir == DOWN:
                action = [0, 0, 1]  # left turn
            elif current_dir == RIGHT:
                action = [1, 0, 0]  # straight
            elif current_dir == LEFT:
                action = [1, 0, 0]  # straight (can't reverse)
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            if current_dir == LEFT:
                action = [0, 1, 0]  # right turn
            elif current_dir == RIGHT:
                action = [0, 0, 1]  # left turn
            elif current_dir == UP:
                action = [1, 0, 0]  # straight
            elif current_dir == DOWN:
                action = [1, 0, 0]  # straight (can't reverse)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if current_dir == LEFT:
                action = [0, 0, 1]  # left turn
            elif current_dir == RIGHT:
                action = [0, 1, 0]  # right turn
            elif current_dir == DOWN:
                action = [1, 0, 0]  # straight
            elif current_dir == UP:
                action = [1, 0, 0]  # straight (can't reverse)
        
        reward, done, score = game.play_step(action)
        
        if done:
            print(f"Game Over! Your Score: {score}")
            game.reset()

if __name__ == '__main__':
    # Choose what to run
    print("AI Snake Game with Deep Q-Learning")
    print("===================================")
    print("1. Train AI Agent")
    print("2. Play with Trained AI")
    print("3. Manual Play")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        plt.ion()  # Turn on interactive mode for plotting
        train()
        plt.ioff()  # Turn off interactive mode
        plt.show()
    elif choice == '2':
        play_trained_model()
    elif choice == '3':
        manual_play()
    else:
        print("Invalid choice. Running training by default...")
        plt.ion()
        train()
        plt.ioff()
        plt.show()