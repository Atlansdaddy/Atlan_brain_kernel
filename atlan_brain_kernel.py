"""
Atlan Brain Kernel - Complete Implementation
A field-based resonance cognitive architecture
Based on the work of Johnathan and Various Ai Assistants
This implements a full artificial brain using resonance propagation
rather than traditional neural networks or transformers.
"""

import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from dataclasses import dataclass, field

# Local configuration
from atlan_kernel_config import KernelConfig

# ---------------------------------------------------------------------
# Global configuration & logger
# ---------------------------------------------------------------------

CONFIG = KernelConfig()

# Configure module-level logger only once
logger = logging.getLogger("atlan_brain_kernel")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, CONFIG.log_level.upper(), logging.INFO))


# ---------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------


class BrainKernelError(Exception):
    """Generic exception raised by the Atlan Brain Kernel."""

    pass


# Global time counter for temporal tracking
global_tick = 0


@dataclass
class CognitiveNode:
    """A basic unit of cognition located at a point in 3-D cognitive space."""

    node_id: int
    position: Tuple[int, int, int]
    context_space: str
    symbolic_anchor: Optional[str] = None

    # Dynamic attributes -------------------------------------------------
    decay_factor: float = field(default_factory=lambda: CONFIG.default_decay_factor)
    resonance_energy: float = 0.0
    threshold: float = field(default_factory=lambda: CONFIG.activation_threshold)
    last_tick: int = 0
    harmonic_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    def distance_to(self, other_node: 'CognitiveNode') -> float:
        """Return Euclidean distance to *other_node*."""
        x1, y1, z1 = self.position
        x2, y2, z2 = other_node.position
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def __repr__(self) -> str:  # type: ignore[override]
        """Readable representation with energy rounded to 4 decimals."""
        return (
            f"Node({self.node_id}, Pos={self.position}, Context={self.context_space}, "
            f"Symbol={self.symbolic_anchor}, Energy={self.resonance_energy:.4f})"
        )


class Nodefield:
    """
    The cognitive substrate - a 3D lattice where nodes exist and
    energy propagates to form thoughts and memories
    """
    
    def __init__(self, initial_size: int = 10):
        self.nodes = {}
        self.size = initial_size
        self.node_counter = 0
        self.memory_chain = []  # temporal experience log
        
    def add_node(self, position: Tuple[int, int, int], context_space: str,
                 symbolic_anchor: Optional[str] = None) -> CognitiveNode:
        """Add a node to the field"""
        node_id = self.node_counter
        node = CognitiveNode(node_id, position, context_space, symbolic_anchor)
        self.nodes[position] = node
        self.node_counter += 1
        return node
        
    def propagate_resonance(self, source_position: Tuple[int, int, int],
                          input_energy: float, importance_weight: float,
                          dampening: float = 0.5) -> List[str]:
        """
        Propagate energy through the field based on distance and importance
        This is the core mechanism of thought formation
        """
        logs = []
        global global_tick
        logger.info(
            "Propagating resonance | tick=%s source_pos=%s input_energy=%.4f importance=%.2f",
            global_tick,
            source_position,
            input_energy,
            importance_weight,
        )
        ε = 0.01  # small constant to avoid division by zero
        
        source_node = self.nodes[source_position]
        weighted_input = input_energy * importance_weight
        source_node.resonance_energy += weighted_input
        source_node.last_tick = global_tick
        
        logs.append(f"Tick {global_tick}: Node {source_node.node_id} "
                   f"'{source_node.symbolic_anchor}' received {weighted_input:.4f} "
                   f"energy (W={importance_weight}). Total: {source_node.resonance_energy:.4f}")
        
        # Log memory event
        self.memory_chain.append((global_tick, None, source_node.symbolic_anchor, weighted_input))
        
        # Propagate if threshold exceeded
        if source_node.resonance_energy >= source_node.threshold:
            for target in self.nodes.values():
                if target == source_node:
                    continue
                    
                distance = source_node.distance_to(target)
                distance_factor = 1 / (distance**2 + ε)
                transferred_energy = weighted_input * dampening * distance_factor
                
                target.resonance_energy += transferred_energy
                target.last_tick = global_tick
                
                logs.append(f"Transferred {transferred_energy:.4f} to Node {target.node_id} "
                           f"'{target.symbolic_anchor}' (Distance={distance:.2f})")
                
                # Log the propagation in memory
                self.memory_chain.append((global_tick, source_node.symbolic_anchor,
                                        target.symbolic_anchor, transferred_energy))
        
        # Apply decay to all nodes
        for node in self.nodes.values():
            before_decay = node.resonance_energy
            node.resonance_energy *= (1 - node.decay_factor)
            logs.append(f"Node {node.node_id} '{node.symbolic_anchor}' decayed "
                       f"from {before_decay:.4f} to {node.resonance_energy:.4f}")
            
        return logs
        
    def snapshot(self) -> List[str]:
        """Get current state of all nodes"""
        return [str(node) for node in self.nodes.values()]


class ReinforcedNodefield(Nodefield):
    """Adds memory replay capabilities for consolidation"""
    
    def replay_memory_chain(self, replay_weight: float = 1.0, 
                          dampening: float = 0.5) -> List[str]:
        """Replay memories to strengthen learning (like sleep/dreams)"""
        logs = []
        global global_tick
        
        logs.append("\n--- REPLAYING MEMORY CHAIN FOR REINFORCEMENT ---")
        
        for tick, source_symbol, target_symbol, energy in self.memory_chain:
            global_tick += 1
            replay_energy = energy * replay_weight
            
            if source_symbol is None:
                # External stimulus
                target_node = next(n for n in self.nodes.values() 
                                 if n.symbolic_anchor == target_symbol)
                target_node.resonance_energy += replay_energy
                target_node.last_tick = global_tick
                logs.append(f"Replay Tick {global_tick}: [External] -> "
                           f"'{target_symbol}' | +{replay_energy:.4f} energy")
            else:
                # Node to node transfer
                source_node = next(n for n in self.nodes.values() 
                                 if n.symbolic_anchor == source_symbol)
                target_node = next(n for n in self.nodes.values() 
                                 if n.symbolic_anchor == target_symbol)
                
                distance = source_node.distance_to(target_node)
                distance_factor = 1 / (distance**2 + 0.01)
                transferred_energy = replay_energy * dampening * distance_factor
                
                target_node.resonance_energy += transferred_energy
                target_node.last_tick = global_tick
                logs.append(f"Replay Tick {global_tick}: '{source_symbol}' -> "
                           f"'{target_symbol}' | +{transferred_energy:.4f} energy")
            
            # Apply decay
            for node in self.nodes.values():
                before = node.resonance_energy
                node.resonance_energy *= (1 - node.decay_factor)
                
        return logs


class AbstractingNodefield(ReinforcedNodefield):
    """Adds abstraction formation from stabilized patterns"""
    
    def derive_abstractions(self, abstraction_threshold: float = 1.0) -> Tuple[List[str], List[str]]:
        """Form abstract concepts from stable energy patterns"""
        logs = []
        abstractions = []
        
        logs.append("\n--- DERIVING ABSTRACTIONS ---")
        
        for node in self.nodes.values():
            if node.resonance_energy >= abstraction_threshold:
                abstraction_label = f"Abstract_{node.symbolic_anchor}"
                abstractions.append(abstraction_label)
                logs.append(f"Node '{node.symbolic_anchor}' stabilized with "
                           f"energy {node.resonance_energy:.4f} → "
                           f"Abstraction: '{abstraction_label}'")
            else:
                logs.append(f"Node '{node.symbolic_anchor}' energy "
                           f"{node.resonance_energy:.4f} below threshold")
                
        return abstractions, logs


class SemanticAnalogicalNodefield(AbstractingNodefield):
    """Adds semantic linking and analogical reasoning"""
    
    def __init__(self, initial_size: int = 10):
        super().__init__(initial_size)
        self.semantic_links = []  # (from_symbol, to_symbol) pairs
        
    def add_semantic_link(self, from_symbol: str, to_symbol: str):
        """Add a semantic relationship between concepts"""
        self.semantic_links.append((from_symbol, to_symbol))
        
    def perform_semantic_analogy_test(self, abstractions: List[str],
                                    similarity_threshold: float = 0.25) -> Tuple[List[str], List[str]]:
        """Find analogies through energy similarity and semantic links"""
        logs = []
        analogies_found = []
        
        logs.append("\n--- SEMANTIC + ANALOGICAL REASONING TEST ---")
        
        # Get energy levels for abstractions
        abstraction_energies = {}
        for abstraction in abstractions:
            symbol = abstraction.replace("Abstract_", "")
            for node in self.nodes.values():
                if node.symbolic_anchor == symbol:
                    abstraction_energies[abstraction] = node.resonance_energy
                    break
        
        # Check energy-based analogies
        for a1 in abstractions:
            for a2 in abstractions:
                if a1 == a2:
                    continue
                energy_diff = abs(abstraction_energies.get(a1, 0) - 
                                abstraction_energies.get(a2, 0))
                if energy_diff <= similarity_threshold:
                    analogy = f"{a1} ~ {a2}"
                    analogies_found.append(analogy)
                    logs.append(f"Analogical match: {analogy} (ΔE={energy_diff:.4f})")
        
        # Check semantic hierarchies
        for from_symbol, to_symbol in self.semantic_links:
            a1 = f"Abstract_{from_symbol}"
            a2 = f"Abstract_{to_symbol}"
            if a1 in abstractions and a2 in abstractions:
                analogy = f"{a1} → {a2} (semantic)"
                analogies_found.append(analogy)
                logs.append(f"Semantic hierarchy: {analogy}")
                
        return analogies_found, logs


class PredictiveNodefield(SemanticAnalogicalNodefield):
    """Adds predictive reasoning from memory chains"""
    
    def build_prediction_model(self) -> List[str]:
        """Learn transition probabilities from experience"""
        logs = []
        logs.append("\n--- BUILDING PREDICTION MODEL ---")
        
        # Count transitions
        transition_counts = {}
        total_counts = {}
        
        for i in range(len(self.memory_chain) - 1):
            _, _, current_symbol, _ = self.memory_chain[i]
            _, _, next_symbol, _ = self.memory_chain[i + 1]
            
            key = (current_symbol, next_symbol)
            transition_counts[key] = transition_counts.get(key, 0) + 1
            total_counts[current_symbol] = total_counts.get(current_symbol, 0) + 1
        
        # Calculate probabilities
        self.prediction_model = {}
        for (from_symbol, to_symbol), count in transition_counts.items():
            probability = count / total_counts[from_symbol]
            self.prediction_model.setdefault(from_symbol, []).append((to_symbol, probability))
            logs.append(f"Learned: '{from_symbol}' → '{to_symbol}' P={probability:.2f}")
            
        return logs
        
    def predict_next(self, current_symbol: str) -> List[str]:
        """Predict likely next symbols"""
        logs = []
        logs.append(f"\n--- PREDICTING FROM '{current_symbol}' ---")
        
        if current_symbol not in self.prediction_model:
            logs.append("No predictions available")
            return logs
            
        predictions = sorted(self.prediction_model[current_symbol], 
                           key=lambda x: -x[1])
        for next_symbol, prob in predictions:
            logs.append(f"Prediction: '{next_symbol}' P={prob:.2f}")
            
        return logs


class ReflectiveNodefield(PredictiveNodefield):
    """Adds self-monitoring and metacognition"""
    
    def assess_belief_stability(self, stability_threshold: float = 1.0) -> List[str]:
        """Monitor internal belief stability"""
        logs = []
        logs.append("\n--- SELF-MONITORING: BELIEF STABILITY ---")
        
        self.stable_beliefs = []
        for node in self.nodes.values():
            if node.resonance_energy >= stability_threshold:
                self.stable_beliefs.append(node.symbolic_anchor)
                logs.append(f"Stable: '{node.symbolic_anchor}' "
                           f"(E={node.resonance_energy:.4f})")
            else:
                logs.append(f"Weak: '{node.symbolic_anchor}' "
                           f"(E={node.resonance_energy:.4f})")
                
        return logs
        
    def reflect_on_predictions(self) -> List[str]:
        """Reflect on prediction consistency"""
        logs = []
        logs.append("\n--- SELF-REFLECTION: PREDICTION CONSISTENCY ---")
        
        for symbol in self.stable_beliefs:
            if symbol in self.prediction_model:
                transitions = self.prediction_model[symbol]
                for to_symbol, prob in transitions:
                    logs.append(f"Belief '{symbol}' expects '{to_symbol}' (P={prob:.2f})")
            else:
                logs.append(f"Belief '{symbol}' has no predictions yet")
                
        return logs


class IntentionNodefield(ReflectiveNodefield):
    """Adds value hierarchies and emergent intentions"""
    
    def __init__(self, initial_size: int = 10):
        super().__init__(initial_size)
        self.reinforcement_log = []
        
    def apply_rewards(self, rewards: Dict[str, float]) -> List[str]:
        """Apply reinforcement to specific concepts"""
        logs = []
        logs.append("\n--- REINFORCEMENT FEEDBACK ---")
        
        for symbol, reward in rewards.items():
            for node in self.nodes.values():
                if node.symbolic_anchor == symbol:
                    before = node.resonance_energy
                    node.resonance_energy += reward
                    after = node.resonance_energy
                    logs.append(f"Reward: '{symbol}' +{reward:.2f} → "
                               f"{before:.4f} → {after:.4f}")
                    self.reinforcement_log.append((symbol, reward, after))
                    break
                    
        return logs
        
    def form_value_hierarchy(self) -> List[str]:
        """Form internal value preferences"""
        logs = []
        logs.append("\n--- VALUE HIERARCHY FORMATION ---")
        
        self.intention_map = {}
        
        for node in self.nodes.values():
            energy = node.resonance_energy
            reward_bonus = sum(r for s, r, _ in self.reinforcement_log 
                             if s == node.symbolic_anchor)
            total_score = energy + reward_bonus
            self.intention_map[node.symbolic_anchor] = total_score
            logs.append(f"'{node.symbolic_anchor}': E={energy:.4f} + "
                       f"Bonus={reward_bonus:.2f} → Score={total_score:.4f}")
                       
        return logs
        
    def select_dominant_intentions(self, top_n: int = 5) -> List[str]:
        """Identify top preferences"""
        logs = []
        logs.append("\n--- DOMINANT INTENTIONS ---")
        
        sorted_prefs = sorted(self.intention_map.items(), key=lambda x: -x[1])
        self.dominant_intentions = sorted_prefs[:top_n]
        
        for symbol, score in self.dominant_intentions:
            logs.append(f"Preference: '{symbol}' → {score:.4f}")
            
        return logs


class CuriosityNodefield(IntentionNodefield):
    """Adds self-directed learning drive"""
    
    def generate_learning_targets(self, exploration_threshold: float = 1.0,
                                curiosity_weight: float = 0.5) -> List[str]:
        """Identify knowledge gaps to explore"""
        logs = []
        logs.append("\n--- SELF-GUIDED LEARNING DRIVE ---")
        
        self.curiosity_targets = []
        
        for node in self.nodes.values():
            if node.symbolic_anchor in self.intention_map:
                score = self.intention_map[node.symbolic_anchor]
                gap = max(exploration_threshold - score, 0)
                curiosity = gap * curiosity_weight
                
                if curiosity > 0:
                    logs.append(f"Curious about '{node.symbolic_anchor}': "
                               f"Gap={gap:.4f}, Curiosity={curiosity:.4f}")
                    self.curiosity_targets.append((node.symbolic_anchor, curiosity))
                    
        return logs
        
    def propose_next_learning_focus(self, top_n: int = 3) -> List[str]:
        """Select top learning priorities"""
        logs = []
        logs.append("\n--- PROPOSED LEARNING TARGETS ---")
        
        sorted_targets = sorted(self.curiosity_targets, key=lambda x: -x[1])
        self.next_learning_goals = sorted_targets[:top_n]
        
        for symbol, score in self.next_learning_goals:
            logs.append(f"Target: '{symbol}' → Curiosity={score:.4f}")
            
        return logs


class FullCognitiveAgent(CuriosityNodefield):
    """
    Complete cognitive agent with all capabilities integrated
    This is the full Atlan Brain Kernel
    """
    
    def __init__(self, initial_size: int = 10):
        super().__init__(initial_size)
        self.conflict_log = []
        self.selected_strategy = None
        self.identity_core = {}
        
    def detect_conflicts(self, conflict_pairs: List[Tuple[str, str]],
                        threshold: float = 0.25) -> List[str]:
        """Detect conflicting beliefs"""
        logs = []
        logs.append("\n--- CONFLICT DETECTION ---")
        
        self.conflict_log = []
        
        for s1, s2 in conflict_pairs:
            e1 = self.get_energy(s1)
            e2 = self.get_energy(s2)
            gap = abs(e1 - e2)
            
            if gap < threshold:
                logs.append(f"Conflict: '{s1}' vs '{s2}' (Δ={gap:.4f})")
                self.conflict_log.append((s1, s2, e1, e2))
            else:
                logs.append(f"No conflict: '{s1}' vs '{s2}'")
                
        return logs
        
    def resolve_conflicts(self) -> List[str]:
        """Resolve conflicts by stability"""
        logs = []
        logs.append("\n--- CONFLICT RESOLUTION ---")
        
        for s1, s2, e1, e2 in self.conflict_log:
            winner = s1 if e1 > e2 else s2
            loser = s2 if e1 > e2 else s1
            logs.append(f"Resolved: '{winner}' over '{loser}'")
            
        return logs
        
    def get_energy(self, symbol: str) -> float:
        """Get energy level of a symbol"""
        for node in self.nodes.values():
            if node.symbolic_anchor == symbol:
                return node.resonance_energy
        return 0.0
        
    def evaluate_strategies(self, strategies: Dict[str, float]) -> List[str]:
        """Evaluate learning strategies"""
        logs = []
        logs.append("\n--- META-STRATEGY REASONING ---")
        
        self.strategy_scores = {}
        
        for name, modifier in strategies.items():
            growth = sum(gap * modifier * random.uniform(0.8, 1.2) 
                        for _, gap in self.curiosity_targets)
            self.strategy_scores[name] = growth
            logs.append(f"Strategy '{name}' → Growth: {growth:.4f}")
            
        return logs
        
    def select_best_strategy(self) -> List[str]:
        """Choose optimal strategy"""
        logs = []
        logs.append("\n--- SELECTED STRATEGY ---")
        
        best = max(self.strategy_scores, key=self.strategy_scores.get)
        self.selected_strategy = best
        logs.append(f"Selected: '{best}' → {self.strategy_scores[best]:.4f}")
        
        return logs
        
    def revise_beliefs(self, revision_threshold: float = 0.5,
                      forgetting_rate: float = 0.25) -> List[str]:
        """Recursive belief revision"""
        logs = []
        logs.append("\n--- BELIEF REVISION ---")
        
        self.revised_beliefs = []
        
        for node in self.nodes.values():
            if node.resonance_energy < revision_threshold:
                before = node.resonance_energy
                node.resonance_energy *= (1 - forgetting_rate)
                logs.append(f"Revising '{node.symbolic_anchor}': "
                           f"{before:.4f} → {node.resonance_energy:.4f}")
                self.revised_beliefs.append(node.symbolic_anchor)
            else:
                logs.append(f"'{node.symbolic_anchor}' stable "
                           f"(E={node.resonance_energy:.4f})")
                           
        return logs
        
    def form_identity_model(self, core_threshold: float = 1.0) -> List[str]:
        """Build persistent identity"""
        logs = []
        logs.append("\n--- IDENTITY MODELING ---")
        
        self.identity_core = {}
        
        for node in self.nodes.values():
            if node.resonance_energy >= core_threshold:
                strength = node.resonance_energy
                self.identity_core[node.symbolic_anchor] = strength
                logs.append(f"Identity Core: '{node.symbolic_anchor}' → {strength:.4f}")
            else:
                logs.append(f"Peripheral: '{node.symbolic_anchor}' "
                           f"(E={node.resonance_energy:.4f})")
                           
        return logs
        
    def run_life_cycle(self, cycles: int = 5) -> List[str]:
        """Run full autonomous life cycles"""
        logs = []
        logs.append("\n--- LIFE-CYCLE SIMULATION ---")
        
        for cycle in range(1, cycles + 1):
            logs.append(f"\n--- Cycle {cycle} ---")
            
            # Update value hierarchy
            self.form_value_hierarchy()
            
            # Self-directed learning
            self.generate_learning_targets(2.5, 0.75)
            self.propose_next_learning_focus(3)
            
            # Apply reinforcement
            targets = {s: 0.5 for s, _ in self.next_learning_goals}
            reward_logs = self.apply_rewards(targets)
            logs.extend(reward_logs)
            
            # Belief revision
            revision_logs = self.revise_beliefs(0.5, 0.25)
            logs.extend(revision_logs)
            
            # Identity update
            identity_logs = self.form_identity_model(1.0)
            logs.extend(identity_logs)
            
        return logs
        
    def visualize_nodefield(self) -> None:
        """Create 3D visualization of the cognitive field"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        for node in self.nodes.values():
            x, y, z = node.position
            size = node.resonance_energy * 100 + 10
            color = 'red' if node.symbolic_anchor in self.identity_core else 'blue'
            alpha = min(node.resonance_energy / 3.0, 1.0)
            
            ax.scatter(x, y, z, s=size, c=color, alpha=alpha)
            if node.symbolic_anchor:
                ax.text(x, y, z, node.symbolic_anchor, fontsize=8)
                
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Atlan Cognitive Nodefield Visualization')
        plt.show()


# Demo functions to test the system
def run_basic_demo():
    """Basic demonstration of core capabilities"""
    print("=== ATLAN BRAIN KERNEL DEMO ===\n")
    
    # Initialize
    brain = FullCognitiveAgent(initial_size=10)
    
    # Create initial knowledge nodes
    print("1. Seeding initial knowledge...")
    biology_nodes = {
        (0, 0, 0): "Plant",
        (1, 0, 0): "Tree", 
        (1, 1, 0): "Apple Tree",
        (1, 1, 1): "Fruit",
        (0, 1, 0): "Food"
    }
    
    physics_nodes = {
        (5, 5, 5): "Force",
        (5, 6, 5): "Motion",
        (5, 5, 6): "Energy",
        (6, 5, 5): "Gravity",
        (5, 6, 6): "Mass"
    }
    
    for pos, symbol in biology_nodes.items():
        brain.add_node(pos, "biology", symbol)
        
    for pos, symbol in physics_nodes.items():
        brain.add_node(pos, "physics", symbol)
        
    # Add semantic links
    brain.add_semantic_link("Apple Tree", "Fruit")
    brain.add_semantic_link("Fruit", "Food")
    
    # Simulate learning events
    print("\n2. Simulating learning events...")
    
    # Biology learning
    global global_tick
    global_tick += 1
    logs = brain.propagate_resonance((1, 1, 0), 5.0, 0.75)  # Apple Tree
    print(f"Learning Apple Tree: {logs[0]}")
    
    global_tick += 1
    logs = brain.propagate_resonance((0, 0, 0), 5.0, 0.5)   # Plant
    print(f"Learning Plant: {logs[0]}")
    
    # Physics learning
    global_tick += 1  
    logs = brain.propagate_resonance((5, 5, 5), 5.0, 1.0)   # Force
    print(f"Learning Force: {logs[0]}")
    
    # Memory consolidation
    print("\n3. Consolidating memories...")
    brain.replay_memory_chain()
    
    # Form abstractions
    print("\n4. Forming abstractions...")
    abstractions, _ = brain.derive_abstractions(1.0)
    print(f"Abstractions formed: {abstractions}")
    
    # Build predictions
    print("\n5. Building predictive model...")
    brain.build_prediction_model()
    pred_logs = brain.predict_next("Apple Tree")
    for log in pred_logs:
        print(log)
        
    # Self-reflection
    print("\n6. Self-monitoring...")
    brain.assess_belief_stability(1.0)
    reflect_logs = brain.reflect_on_predictions()
    for log in reflect_logs[:3]:
        print(log)
        
    # Apply rewards
    print("\n7. Applying reinforcement...")
    rewards = {
        "Fruit": 1.0,
        "Mass": 0.75,
        "Energy": 0.5
    }
    brain.apply_rewards(rewards)
    
    # Form identity
    print("\n8. Forming identity...")
    brain.form_value_hierarchy()
    brain.form_identity_model(1.0)
    print(f"Identity core: {list(brain.identity_core.keys())}")
    
    # Life cycle
    print("\n9. Running life cycles...")
    life_logs = brain.run_life_cycle(3)
    print(f"Completed 3 life cycles. Final identity: {list(brain.identity_core.keys())}")
    
    # Visualize
    print("\n10. Visualizing cognitive field...")
    brain.visualize_nodefield()
    
    return brain


def run_advanced_demo():
    """Advanced demonstration with conflict resolution and meta-strategies"""
    print("\n=== ADVANCED ATLAN DEMO ===\n")
    
    brain = run_basic_demo()
    
    # Conflict detection
    print("\n11. Detecting conflicts...")
    conflicts = [("Fruit", "Food"), ("Plant", "Tree"), ("Energy", "Motion")]
    conflict_logs = brain.detect_conflicts(conflicts, 0.5)
    for log in conflict_logs:
        print(log)
        
    resolve_logs = brain.resolve_conflicts()
    for log in resolve_logs:
        print(log)
        
    # Meta-strategy evaluation
    print("\n12. Evaluating learning strategies...")
    strategies = {
        "Exploit Known": 0.6,
        "Balanced": 1.0,
        "Explore Unknown": 1.4
    }
    brain.evaluate_strategies(strategies)
    brain.select_best_strategy()
    print(f"Selected strategy: {brain.selected_strategy}")
    
    # Extended life cycle
    print("\n13. Extended autonomous evolution...")
    brain.run_life_cycle(5)
    
    # Final state
    print("\n14. Final cognitive state:")
    print(f"Stable beliefs: {brain.stable_beliefs}")
    print(f"Identity core: {list(brain.identity_core.keys())}")
    print(f"Dominant intentions: {[s for s, _ in brain.dominant_intentions[:3]]}")
    
    return brain


if __name__ == "__main__":
    # Run demonstrations
    print("Starting Atlan Brain Kernel...")
    print("This implements a field-based cognitive architecture")
    print("based on resonance propagation rather than neural networks.\n")
    
    # Basic demo
    brain = run_basic_demo()
    
    # Advanced demo
    # brain = run_advanced_demo()
    
    print("\nDemo complete!")
    print("The Atlan Brain Kernel is a novel approach to AGI")
    print("that models cognition as emergent field dynamics.")
