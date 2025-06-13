#!/bin/bash
# Atlan Brain Kernel - Quick Start Script
# This script sets up the complete development environment

echo "==================================="
echo "Atlan Brain Kernel - Quick Setup"
echo "==================================="
echo ""

# Create project directory
PROJECT_NAME="atlan-brain-kernel"
echo "Creating project directory: $PROJECT_NAME"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create subdirectories
echo "Creating project structure..."
mkdir -p experiments
mkdir -p visualizations
mkdir -p docs/tutorials

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << EOF
numpy>=1.19.0
matplotlib>=3.3.0
pytest>=6.0
plotly>=5.0
networkx>=2.5
psutil>=5.8.0
EOF

# Create __init__.py files
touch experiments/__init__.py

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << EOF
# Python
*.pyc
__pycache__/
*.egg-info/
dist/
build/
.pytest_cache/

# Visualizations
visualizations/*.png
visualizations/*.jpg
visualizations/*.pdf

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb
EOF

# Create MIT License
echo "Creating LICENSE..."
cat > LICENSE << EOF
MIT License

Copyright (c) 2024 Atlan Brain Kernel Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create a simple run script
echo "Creating run.py helper script..."
cat > run.py << 'EOF'
#!/usr/bin/env python3
"""
Atlan Brain Kernel - Interactive Runner
"""

import sys
import os

def main():
    print("Atlan Brain Kernel - Interactive Runner")
    print("=====================================\n")
    
    print("Choose an option:")
    print("1. Run basic demo")
    print("2. Run all examples")
    print("3. Run tests")
    print("4. Create new experiment")
    print("5. Visualize cognitive field")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        print("\nRunning basic demo...")
        os.system("python atlan_brain_kernel.py")
    elif choice == "2":
        print("\nRunning all examples...")
        os.system("python examples.py")
    elif choice == "3":
        print("\nRunning tests...")
        os.system("pytest test_atlan.py -v")
    elif choice == "4":
        create_experiment()
    elif choice == "5":
        visualize_field()
    elif choice == "6":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice!")
        
def create_experiment():
    """Create a new experiment file"""
    name = input("Enter experiment name: ")
    filename = f"experiments/{name}.py"
    
    template = '''"""
Experiment: {name}
Description: Add your description here
"""

from atlan_brain_kernel import FullCognitiveAgent

def run_experiment():
    # Create brain
    brain = FullCognitiveAgent()
    
    # Add your experiment code here
    
    return brain

if __name__ == "__main__":
    brain = run_experiment()
    brain.visualize_nodefield()
'''.format(name=name)
    
    with open(filename, 'w') as f:
        f.write(template)
        
    print(f"\nCreated {filename}")
    print("Edit the file to add your experiment code")
    
def visualize_field():
    """Quick visualization"""
    code = '''
from atlan_brain_kernel import FullCognitiveAgent

brain = FullCognitiveAgent()

# Quick setup
for i in range(5):
    brain.add_node((i, i, i), "test", f"Concept{i}")
    brain.propagate_resonance((i, i, i), 5.0, 0.5 + i*0.1)

brain.visualize_nodefield()
'''
    exec(code)

if __name__ == "__main__":
    main()
EOF

# Create virtual environment setup script
echo "Creating venv_setup.sh..."
cat > venv_setup.sh << 'EOF'
#!/bin/bash
# Setup virtual environment for Atlan

echo "Setting up Python virtual environment..."

# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo "To activate: source venv/bin/activate"
EOF

chmod +x venv_setup.sh

# Create a Makefile for common tasks
echo "Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help install test run clean docs

help:
	@echo "Atlan Brain Kernel - Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run interactive demo"
	@echo "  make clean      - Clean generated files"
	@echo "  make docs       - Generate documentation"

install:
	pip install -r requirements.txt

test:
	pytest test_atlan.py -v

run:
	python run.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist build

docs:
	@echo "Generating documentation..."
	python -m pydoc -w atlan_brain_kernel
EOF

# Final setup instructions
echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Copy the Python files into this directory:"
echo "   - atlan_brain_kernel.py"
echo "   - test_atlan.py"
echo "   - examples.py"
echo "   - setup.py"
echo ""
echo "2. Set up virtual environment (optional):"
echo "   ./venv_setup.sh"
echo "   source venv/bin/activate"
echo ""
echo "3. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Run the demo:"
echo "   python atlan_brain_kernel.py"
echo ""
echo "5. Or use the interactive runner:"
echo "   python run.py"
echo ""
echo "Happy experimenting with Atlan!"
