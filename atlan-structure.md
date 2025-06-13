# Atlan Brain Kernel - Project Structure

## Complete Setup Instructions for Cursor

### 1. Create Project Directory

```bash
mkdir atlan-brain-kernel
cd atlan-brain-kernel
```

### 2. File Structure

Create the following structure:

```
atlan-brain-kernel/
│
├── README.md                 # Main documentation
├── setup.py                  # Installation script
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
│
├── atlan_brain_kernel.py     # Main implementation
├── test_atlan.py            # Unit tests
├── examples.py              # Usage examples
│
├── experiments/             # Your experiments
│   ├── __init__.py
│   └── custom_domains.py    # Custom domain experiments
│
├── visualizations/          # Output visualizations
│   └── .gitkeep
│
└── docs/                    # Additional documentation
    ├── theory.md            # Theoretical background
    ├── api.md              # API reference
    └── tutorials/          # Step-by-step tutorials
        ├── 01_basics.md
        ├── 02_memory.md
        └── 03_advanced.md
```

### 3. Create requirements.txt

```txt
numpy>=1.19.0
matplotlib>=3.3.0
pytest>=6.0
```

### 4. Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic demo
python atlan_brain_kernel.py

# Run all examples
python examples.py

# Run tests
pytest test_atlan.py -v

# Install as package (optional)
pip install -e .
```

### 5. Cursor IDE Setup

1. Open the project folder in Cursor
2. Install recommended extensions:
   - Python
   - Pylance
   - Python Test Explorer

3. Configure settings.json:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

### 6. First Experiment Template

Create `experiments/my_first_brain.py`:

```python
from atlan_brain_kernel import FullCognitiveAgent

def create_my_brain():
    """Create a custom cognitive agent"""
    brain = FullCognitiveAgent()
    
    # Add your domain knowledge
    my_concepts = {
        (0, 0, 0): ("personal", "Self"),
        (1, 0, 0): ("personal", "Goals"),
        (2, 0, 0): ("personal", "Values"),
        # Add more...
    }
    
    for pos, (domain, concept) in my_concepts.items():
        brain.add_node(pos, domain, concept)
    
    # Add relationships
    brain.add_semantic_link("Goals", "Values")
    
    # Train
    brain.propagate_resonance((0, 0, 0), 5.0, 1.0)
    
    # Evolve
    brain.run_life_cycle(cycles=10)
    
    return brain

if __name__ == "__main__":
    my_brain = create_my_brain()
    my_brain.visualize_nodefield()
```

### 7. Git Setup (Optional)

```bash
git init
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "visualizations/*.png" >> .gitignore
echo ".pytest_cache/" >> .gitignore
git add .
git commit -m "Initial Atlan Brain Kernel implementation"
```

### 8. Advanced Configuration

For production use, create `config.py`:

```python
# Atlan Configuration
DEFAULT_DECAY_FACTOR = 0.1
DEFAULT_THRESHOLD = 1.0
DEFAULT_DAMPENING = 0.5
DEFAULT_GRID_SIZE = 10

# Importance weights
IMPORTANCE_WEIGHTS = {
    "survival_critical": 1.0,
    "mission_critical": 0.75,
    "medium": 0.5,
    "low": 0.25,
    "noise": 0.05
}

# Learning strategies
LEARNING_STRATEGIES = {
    "exploit": 0.6,
    "balanced": 1.0,
    "explore": 1.4
}
```

### 9. Debugging in Cursor

Set breakpoints in interesting places:
- Line where energy propagates in `propagate_resonance()`
- Abstraction formation in `derive_abstractions()`
- Identity formation in `form_identity_model()`

### 10. Performance Monitoring

Add this to track performance:

```python
import time
import psutil
import matplotlib.pyplot as plt

def monitor_performance(brain, cycles=10):
    """Monitor CPU and memory during life cycles"""
    cpu_usage = []
    memory_usage = []
    
    process = psutil.Process()
    
    for i in range(cycles):
        start_time = time.time()
        
        # Measure before
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run cycle
        brain.run_life_cycle(cycles=1)
        
        # Measure after
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        cpu_usage.append((cpu_before + cpu_after) / 2)
        memory_usage.append(memory_after)
        
        elapsed = time.time() - start_time
        print(f"Cycle {i+1}: {elapsed:.2f}s, "
              f"CPU: {cpu_usage[-1]:.1f}%, "
              f"Memory: {memory_usage[-1]:.1f}MB")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(cpu_usage)
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('Performance Monitoring')
    
    ax2.plot(memory_usage)
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_xlabel('Cycle')
    
    plt.tight_layout()
    plt.savefig('visualizations/performance.png')
    plt.show()
```

## Next Steps

1. **Experiment**: Modify parameters and observe changes
2. **Extend**: Add new domains and concepts
3. **Visualize**: Create custom visualization functions
4. **Optimize**: Profile and improve performance
5. **Document**: Keep notes on interesting emergent behaviors

## Troubleshooting

**ImportError**: Make sure you're in the project directory
**Memory Issues**: Reduce grid size or number of nodes
**Slow Performance**: Decrease life cycle count or use sparse nodefields

## Support

- Check the examples.py file for usage patterns
- Run tests to verify your changes
- Use the visualization tools to understand field dynamics

Remember: This is experimental cognitive architecture. Document any interesting emergent behaviors you observe!