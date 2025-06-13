"""
Atlan Brain Kernel - Usage Examples
Demonstrates various capabilities of the cognitive architecture
"""

from atlan_brain_kernel import FullCognitiveAgent, global_tick
import matplotlib.pyplot as plt


def example_1_basic_learning():
    """Example 1: Basic concept learning and association"""
    print("\n=== Example 1: Basic Learning ===")
    
    brain = FullCognitiveAgent()
    
    # Create a simple knowledge structure
    brain.add_node((0, 0, 0), "animals", "Animal")
    brain.add_node((1, 0, 0), "animals", "Mammal")
    brain.add_node((2, 0, 0), "animals", "Dog")
    brain.add_node((1, 1, 0), "animals", "Bird")
    brain.add_node((2, 1, 0), "animals", "Eagle")
    
    # Add semantic relationships
    brain.add_semantic_link("Dog", "Mammal")
    brain.add_semantic_link("Mammal", "Animal")
    brain.add_semantic_link("Eagle", "Bird") 
    brain.add_semantic_link("Bird", "Animal")
    
    # Simulate learning about dogs
    global global_tick
    global_tick = 1
    print("\nLearning about 'Dog' (high importance)...")
    brain.propagate_resonance((2, 0, 0), input_energy=5.0, importance_weight=0.9)
    
    # Check what got activated
    print("\nActivated concepts:")
    for node in brain.nodes.values():
        if node.resonance_energy > 0.1:
            print(f"  {node.symbolic_anchor}: {node.resonance_energy:.2f}")
            
    return brain


def example_2_memory_consolidation():
    """Example 2: Memory consolidation through replay"""
    print("\n=== Example 2: Memory Consolidation ===")
    
    brain = example_1_basic_learning()
    
    print("\nBefore consolidation:")
    for node in brain.nodes.values():
        print(f"  {node.symbolic_anchor}: {node.resonance_energy:.2f}")
        
    # Consolidate memories (like sleep)
    print("\nConsolidating memories...")
    brain.replay_memory_chain(replay_weight=0.8)
    
    print("\nAfter consolidation:")
    for node in brain.nodes.values():
        if node.resonance_energy > 0.1:
            print(f"  {node.symbolic_anchor}: {node.resonance_energy:.2f}")
            
    return brain


def example_3_abstraction_formation():
    """Example 3: Forming abstract concepts"""
    print("\n=== Example 3: Abstraction Formation ===")
    
    brain = FullCognitiveAgent()
    
    # Create concepts at different levels
    concepts = {
        "concrete": [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        "abstract": [(0, 1, 1), (1, 1, 1), (2, 1, 1)]
    }
    
    # Add concrete objects
    for i, pos in enumerate(concepts["concrete"]):
        brain.add_node(pos, "objects", f"Object{i}")
        
    # Add abstract concepts
    abstract_names = ["Shape", "Color", "Size"]
    for i, pos in enumerate(concepts["abstract"]):
        brain.add_node(pos, "properties", abstract_names[i])
        
    # Learn with different intensities
    global global_tick
    
    # Strong learning for abstract concepts
    for i, pos in enumerate(concepts["abstract"]):
        global_tick += 1
        brain.propagate_resonance(pos, 5.0, 1.0)
        
    # Weak learning for concrete objects
    for i, pos in enumerate(concepts["concrete"]):
        global_tick += 1
        brain.propagate_resonance(pos, 5.0, 0.3)
        
    # Form abstractions
    abstractions, _ = brain.derive_abstractions(abstraction_threshold=1.0)
    
    print("\nFormed abstractions:")
    for abstraction in abstractions:
        print(f"  {abstraction}")
        
    return brain


def example_4_prediction():
    """Example 4: Learning sequences and making predictions"""
    print("\n=== Example 4: Predictive Learning ===")
    
    brain = FullCognitiveAgent()
    
    # Create a sequence: Morning -> Coffee -> Work -> Lunch
    sequence = [
        ((0, 0, 0), "Morning"),
        ((1, 0, 0), "Coffee"),
        ((2, 0, 0), "Work"),
        ((3, 0, 0), "Lunch")
    ]
    
    for pos, concept in sequence:
        brain.add_node(pos, "routine", concept)
        
    # Learn the sequence multiple times
    global global_tick
    
    print("\nLearning daily routine...")
    for _ in range(3):  # Repeat 3 times
        for pos, _ in sequence:
            global_tick += 1
            brain.propagate_resonance(pos, 3.0, 0.7)
            
    # Build prediction model
    brain.build_prediction_model()
    
    # Make predictions
    print("\nPredictions:")
    for _, concept in sequence[:-1]:
        predictions = brain.predict_next(concept)
        print(f"\nAfter {concept}:")
        for pred in predictions[1:]:  # Skip header
            print(f"  {pred}")
            
    return brain


def example_5_conflict_resolution():
    """Example 5: Handling conflicting information"""
    print("\n=== Example 5: Conflict Resolution ===")
    
    brain = FullCognitiveAgent()
    
    # Create potentially conflicting beliefs
    brain.add_node((0, 0, 0), "facts", "EarthRound")
    brain.add_node((1, 0, 0), "facts", "EarthFlat")
    brain.add_node((0, 1, 0), "facts", "SunCenter")
    brain.add_node((1, 1, 0), "facts", "EarthCenter")
    
    # Learn with different confidence levels
    global global_tick
    
    # Strong evidence for correct beliefs
    global_tick += 1
    brain.propagate_resonance((0, 0, 0), 10.0, 1.0)  # Earth is round
    global_tick += 1
    brain.propagate_resonance((0, 1, 0), 8.0, 0.9)   # Sun at center
    
    # Weak evidence for incorrect beliefs
    global_tick += 1
    brain.propagate_resonance((1, 0, 0), 2.0, 0.2)   # Earth is flat
    global_tick += 1
    brain.propagate_resonance((1, 1, 0), 3.0, 0.3)   # Earth at center
    
    # Detect conflicts
    conflicts = [
        ("EarthRound", "EarthFlat"),
        ("SunCenter", "EarthCenter")
    ]
    
    print("\nDetecting conflicts...")
    brain.detect_conflicts(conflicts, threshold=0.5)
    
    print("\nResolving conflicts...")
    resolution_logs = brain.resolve_conflicts()
    for log in resolution_logs:
        print(f"  {log}")
        
    return brain


def example_6_multidomain_learning():
    """Example 6: Learning across multiple domains"""
    print("\n=== Example 6: Multi-Domain Learning ===")
    
    brain = FullCognitiveAgent()
    
    # Domain 1: Mathematics
    math_concepts = {
        (0, 0, 0): "Number",
        (1, 0, 0): "Addition",
        (2, 0, 0): "Multiplication"
    }
    
    # Domain 2: Language
    language_concepts = {
        (5, 5, 5): "Word",
        (6, 5, 5): "Sentence",
        (7, 5, 5): "Paragraph"
    }
    
    # Domain 3: Music
    music_concepts = {
        (10, 10, 10): "Note",
        (11, 10, 10): "Chord",
        (12, 10, 10): "Melody"
    }
    
    # Add all concepts
    for pos, concept in math_concepts.items():
        brain.add_node(pos, "mathematics", concept)
        
    for pos, concept in language_concepts.items():
        brain.add_node(pos, "language", concept)
        
    for pos, concept in music_concepts.items():
        brain.add_node(pos, "music", concept)
        
    # Learn different domains with different intensities
    global global_tick
    
    print("\nLearning mathematics...")
    for pos in math_concepts:
        global_tick += 1
        brain.propagate_resonance(pos, 5.0, 0.8)
        
    print("Learning language...")
    for pos in language_concepts:
        global_tick += 1
        brain.propagate_resonance(pos, 5.0, 0.6)
        
    print("Learning music...")
    for pos in music_concepts:
        global_tick += 1
        brain.propagate_resonance(pos, 5.0, 0.4)
        
    # Form value hierarchy
    brain.form_value_hierarchy()
    brain.select_dominant_intentions(top_n=5)
    
    print("\nDominant knowledge areas:")
    for concept, score in brain.dominant_intentions:
        print(f"  {concept}: {score:.2f}")
        
    return brain


def example_7_autonomous_growth():
    """Example 7: Autonomous cognitive development"""
    print("\n=== Example 7: Autonomous Growth ===")
    
    brain = FullCognitiveAgent()
    
    # Seed initial knowledge
    initial_concepts = {
        (0, 0, 0): ("science", "Physics"),
        (1, 0, 0): ("science", "Chemistry"),
        (2, 0, 0): ("science", "Biology"),
        (0, 1, 0): ("art", "Painting"),
        (1, 1, 0): ("art", "Music"),
        (2, 1, 0): ("art", "Poetry"),
        (0, 2, 0): ("skill", "Coding"),
        (1, 2, 0): ("skill", "Writing"),
        (2, 2, 0): ("skill", "Speaking")
    }
    
    for pos, (domain, concept) in initial_concepts.items():
        brain.add_node(pos, domain, concept)
        
    # Initial learning
    global global_tick
    
    print("\nInitial learning phase...")
    for i, pos in enumerate(initial_concepts.keys()):
        global_tick += 1
        # Varying importance
        importance = 0.3 + (i * 0.1) % 0.7
        brain.propagate_resonance(pos, 5.0, importance)
        
    # Run autonomous life cycles
    print("\nRunning autonomous development...")
    brain.run_life_cycle(cycles=3)
    
    # Check final identity
    print("\nFinal cognitive identity:")
    sorted_identity = sorted(brain.identity_core.items(), 
                           key=lambda x: -x[1])
    for concept, strength in sorted_identity[:5]:
        print(f"  {concept}: {strength:.2f}")
        
    # Check learning targets
    brain.generate_learning_targets(exploration_threshold=2.0)
    brain.propose_next_learning_focus(top_n=3)
    
    print("\nSelf-identified learning needs:")
    for concept, curiosity in brain.next_learning_goals:
        print(f"  {concept}: curiosity={curiosity:.2f}")
        
    return brain


def example_8_meta_learning():
    """Example 8: Learning how to learn better"""
    print("\n=== Example 8: Meta-Learning ===")
    
    brain = example_7_autonomous_growth()
    
    # Evaluate different learning strategies
    strategies = {
        "Depth-First": 1.5,      # Deep dive into few topics
        "Breadth-First": 0.8,    # Shallow coverage of many topics
        "Balanced": 1.0,         # Mix of both
        "Curiosity-Driven": 1.3  # Follow interests
    }
    
    print("\nEvaluating learning strategies...")
    brain.evaluate_strategies(strategies)
    
    print("\nStrategy evaluation results:")
    for strategy, score in brain.strategy_scores.items():
        print(f"  {strategy}: {score:.2f}")
        
    brain.select_best_strategy()
    print(f"\nSelected strategy: {brain.selected_strategy}")
    
    return brain


def visualize_cognitive_evolution():
    """Visualize how the cognitive field evolves over time"""
    print("\n=== Cognitive Evolution Visualization ===")
    
    brain = FullCognitiveAgent()
    
    # Create a grid of concepts
    grid_size = 3
    concepts = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                concept = f"C{x}{y}{z}"
                brain.add_node((x*2, y*2, z*2), "general", concept)
                concepts.append(((x*2, y*2, z*2), concept))
                
    # Track energy over time
    energy_history = {concept: [] for _, concept in concepts}
    
    # Simulate learning over time
    global global_tick
    timesteps = 20
    
    for t in range(timesteps):
        # Random learning events
        import random
        pos, _ = random.choice(concepts)
        importance = random.uniform(0.3, 1.0)
        
        global_tick += 1
        brain.propagate_resonance(pos, 5.0, importance)
        
        # Record energy levels
        for node_pos, concept in concepts:
            energy = brain.nodes[node_pos].resonance_energy
            energy_history[concept].append(energy)
            
        # Occasional consolidation
        if t % 5 == 0:
            brain.replay_memory_chain(replay_weight=0.7)
            
    # Plot evolution
    plt.figure(figsize=(12, 8))
    
    # Plot energy evolution for some concepts
    sample_concepts = random.sample(list(energy_history.keys()), 
                                   min(8, len(concepts)))
    
    for concept in sample_concepts:
        plt.plot(energy_history[concept], label=concept, alpha=0.7)
        
    plt.xlabel('Time Steps')
    plt.ylabel('Resonance Energy')
    plt.title('Cognitive Field Evolution Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Final 3D visualization
    brain.visualize_nodefield()
    
    return brain


# Run all examples
if __name__ == "__main__":
    print("Atlan Brain Kernel - Examples")
    print("="*50)
    
    # Run examples
    brain1 = example_1_basic_learning()
    brain2 = example_2_memory_consolidation()
    brain3 = example_3_abstraction_formation()
    brain4 = example_4_prediction()
    brain5 = example_5_conflict_resolution()
    brain6 = example_6_multidomain_learning()
    brain7 = example_7_autonomous_growth()
    brain8 = example_8_meta_learning()
    
    # Visualization
    print("\n" + "="*50)
    visualize_cognitive_evolution()
    
    print("\n✓ All examples completed!")
    print("\nThe Atlan Brain Kernel demonstrates:")
    print("- Field-based resonance propagation")
    print("- Emergent memory formation")
    print("- Self-organizing abstractions")
    print("- Predictive reasoning")
    print("- Conflict resolution")
    print("- Multi-domain integration")
    print("- Autonomous learning")
    print("- Meta-cognitive adaptation")