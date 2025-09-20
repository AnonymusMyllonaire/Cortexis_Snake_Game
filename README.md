# 🐍 AI Snake Game with Deep Q-Learning

An intelligent Snake game where an AI agent learns to play using Deep Reinforcement Learning (Deep Q-Learning). Watch as the AI evolves from random movements to strategic gameplay!

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)

## 🎯 Project Overview

This project implements a classic Snake game with an AI agent that learns to play autonomously using Deep Q-Learning reinforcement learning. The AI starts with random movements and gradually develops sophisticated strategies through trial and error.

### Key Features

- **Classic Snake Game**: Built with Pygame for smooth gameplay
- **AI Agent**: Deep Q-Network (DQN) implementation using PyTorch
- **Real-time Training**: Watch the AI learn and improve in real-time
- **Multiple Modes**: Train AI, play with trained model, or play manually
- **Performance Tracking**: Real-time plots showing learning progress
- **Model Persistence**: Save and load trained models

## 🧠 How It Works

### Game Environment
- **Grid-based Movement**: Snake moves on a 32x32 grid
- **Food System**: Randomly spawned food increases score and snake length
- **Collision Detection**: Game ends on wall or self-collision
- **Score Tracking**: Points awarded for eating food

### AI Implementation
- **State Space**: 11-dimensional state vector including:
  - Danger detection (straight, left, right directions)
  - Current movement direction (up, down, left, right)
  - Food position relative to snake head
- **Action Space**: 3 possible actions (straight, turn left, turn right)
- **Neural Network**: Fully connected network with 256 hidden units
- **Training Algorithm**: Deep Q-Learning with experience replay

### Reward System
- **+10 points**: Snake eats food
- **-10 points**: Game over (collision)
- **-1 point**: Each move (encourages efficiency)

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the game**:
   ```bash
   python snake_ai.py
   ```

### Usage Options

When you run the program, you'll see three options:

```
AI Snake Game with Deep Q-Learning
===================================
1. Train AI Agent
2. Play with Trained AI
3. Manual Play

Enter your choice (1-3):
```

#### Option 1: Train AI Agent
- Starts training from scratch
- Shows real-time learning progress
- Automatically saves the best model as `model.pth`
- Displays training plots every 50 games
- Training stops after 500 games or when performance goals are met

#### Option 2: Play with Trained AI
- Loads a previously trained model (`model.pth`)
- Watch the AI play autonomously
- No learning occurs, pure performance mode
- Requires a trained model file to exist

#### Option 3: Manual Play
- Control the snake yourself using keyboard
- Use WASD or Arrow keys to change direction
- Good for testing the game mechanics
- Compare your performance with the AI

## 📊 Training Process

### Learning Stages

1. **Random Exploration (Games 1-50)**
   - AI makes random moves
   - Scores typically 0-5 points
   - High exploration, low performance

2. **Basic Learning (Games 50-150)**
   - AI starts recognizing patterns
   - Learns basic survival strategies
   - Scores improve to 5-15 points

3. **Strategic Development (Games 150-300)**
   - Develops food-seeking behavior
   - Better collision avoidance
   - Scores reach 15-30 points

4. **Optimization (Games 300+)**
   - Fine-tunes strategies
   - Efficient pathfinding
   - Can achieve scores of 50+ points

### Performance Metrics

The training displays:
- **Current Game Score**: Points earned in the current game
- **Record Score**: Best score achieved so far
- **Average Score**: Mean performance over all games
- **Real-time Plots**: Visual representation of learning progress

## 🎮 Controls (Manual Play Mode)

| Key | Action |
|-----|--------|
| W / ↑ | Change direction up |
| S / ↓ | Change direction down |
| A / ← | Change direction left |
| D / → | Change direction right |

*Note: Snake cannot reverse direction directly*

## 🔧 Configuration

### Adjusting Game Speed
In the `play_step()` method, modify:
```python
self.clock.tick(10)  # Change number for different speeds
```
- **Slower**: Use lower numbers (5, 3)
- **Faster**: Use higher numbers (15, 30)

### Modifying Neural Network
In the `DQN` class:
```python
self.linear1 = nn.Linear(input_size, hidden_size)  # Adjust hidden_size
```

### Changing Training Parameters
In the `DQNAgent` class:
```python
self.gamma = 0.9     # Discount factor
self.epsilon = 80    # Initial exploration rate
lr = 0.001          # Learning rate
```

## 📁 Project Structure

```
snake-ai/
│
├── snake_ai.py          # Main game and AI implementation
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── model.pth           # Saved AI model (generated after training)
└── training_plots/     # Training progress plots (optional)
```

## 🔬 Technical Details

### Deep Q-Learning Algorithm
- **Q-Network**: Approximates Q-values for state-action pairs
- **Target Network**: Stabilizes training (implicit in this implementation)
- **Experience Replay**: Stores and samples past experiences
- **Epsilon-Greedy**: Balances exploration vs exploitation

### State Representation
The AI receives an 11-dimensional state vector:
```python
[danger_straight, danger_right, danger_left,
 direction_left, direction_right, direction_up, direction_down,
 food_left, food_right, food_up, food_down]
```

### Network Architecture
```
Input Layer (11 neurons) → 
Hidden Layer 1 (256 neurons) → ReLU →
Hidden Layer 2 (256 neurons) → ReLU →
Output Layer (3 neurons)
```

## 📈 Expected Results

### Training Performance
- **Initial Phase**: Random performance, scores 0-5
- **Learning Phase**: Gradual improvement, scores 10-25
- **Mastery Phase**: Strategic play, scores 30-100+

### Trained AI Capabilities
- Efficient pathfinding to food
- Collision avoidance strategies
- Optimal space utilization
- Long-term survival planning

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'pygame'"**
```bash
pip install pygame
```

**"No trained model found"**
- Run training mode first (Option 1)
- Ensure `model.pth` exists in the same directory

**Game runs too fast/slow**
- Modify `self.clock.tick(10)` in the code
- Lower numbers = slower, higher numbers = faster

**Training not improving**
- Let it run longer (AI needs 100+ games to show improvement)
- Check that rewards are being calculated correctly
- Ensure sufficient exploration (epsilon value)

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the neural network architecture
- Adding new features (different game modes, better graphics)
- Optimizing training performance
- Adding more sophisticated AI algorithms
- Enhancing documentation

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Educational Value

This project demonstrates:
- **Reinforcement Learning**: Q-learning algorithm implementation
- **Neural Networks**: Deep learning with PyTorch
- **Game Development**: Using Pygame for interactive applications
- **Python Programming**: Object-oriented design and best practices
- **Data Visualization**: Real-time plotting with Matplotlib

Perfect for students learning AI, machine learning, or game development!

## 🔮 Future Enhancements

Potential improvements:
- **Double DQN**: More stable training
- **Dueling DQN**: Better value estimation
- **Prioritized Experience Replay**: More efficient learning
- **CNN-based State**: Use raw pixels as input
- **Multi-agent Training**: Multiple snakes competing
- **Web Interface**: Browser-based gameplay
- **Advanced Graphics**: Better visual effects

---

**Happy Learning! 🎉**

*Watch your AI evolve from a confused snake into a strategic master!*
