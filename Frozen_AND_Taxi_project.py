# ========================================================================
# 1. Import Dependencies
# ========================================================================
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time  # For controlling animation speed

# ========================================================================
# 2. Enhanced Training Function with Visualization
# ========================================================================
def train_agent(episodes=5000, is_slippery=True, render_training=False):
    """Train the AI with live environment visualization"""
    
    # ====================================================================
    # 3. Environment Setup with Rendering
    # ====================================================================
    env = gym.make('FrozenLake-v1',
                 map_name="8x8",
                 is_slippery=is_slippery,
                 render_mode='human' if render_training else None)
    
    # ====================================================================
    # 4. Q-Table Initialization
    # ====================================================================
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    # ====================================================================
    # 5. Learning Parameters
    # ====================================================================
    learning_rate = 0.8
    discount = 0.95
    epsilon = 1.0
    epsilon_decay = 0.001
    rng = np.random.default_rng()
    
    # ====================================================================
    # 6. Training Loop with Live Visualization
    # ====================================================================
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0
        
        # Show initial state
        if render_training and episode % 100 == 0:  # Render every 100th episode
            env.render()
            time.sleep(0.1)
        
        while not done:
            # Choose action
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            
            # Perform action
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-table
            q_table[state, action] += learning_rate * (
                reward + discount * np.max(q_table[new_state, :]) - q_table[state, action]
            )
            
            # Visual update
            if render_training and episode % 100 == 0:
                env.render()
                time.sleep(0.1)  # Slow down for visibility
                
            state = new_state
            steps += 1
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: Steps {steps}, Success {reward == 1}")
            
        # Decay exploration rate
        epsilon = max(epsilon - epsilon_decay, 0.01)
    
    env.close()
    return q_table

# ========================================================================
# 7. Live Demonstration Function
# ========================================================================
def demonstrate_agent(q_table, is_slippery=True):
    """Show trained agent performing 5 complete runs"""
    env = gym.make('FrozenLake-v1',
                 map_name="8x8",
                 is_slippery=is_slippery,
                 render_mode='human')
    
    for _ in range(5):  # Demonstrate 5 times
        state = env.reset()[0]
        done = False
        while not done:
            action = np.argmax(q_table[state, :])
            state, _, done, _ = env.step(action)
            env.render()
            time.sleep(0.5)  # Slow motion for clarity
        print("Goal reached!" if reward == 1 else "Failed!")
        time.sleep(1)
    
    env.close()

# ========================================================================
# 8. Main Execution Flow
# ========================================================================
if __name__ == '__main__':
    # Phase 1: Training with occasional visualization
    print("Training with slippery surface...")
    slippery_q = train_agent(episodes=5000, 
                           is_slippery=True, 
                           render_training=True)  # See occasional training
    
    # Phase 2: Final demonstration
    print("\nShowing trained agent in action...")
    demonstrate_agent(slippery_q, is_slippery=True)
    
    # Save results
    with open('slippery_q.pkl', 'wb') as f:
        pickle.dump(slippery_q, f)