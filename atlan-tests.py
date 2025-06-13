"""
Unit tests for Atlan Brain Kernel
"""

import pytest
import numpy as np
from atlan_brain_kernel import (
    CognitiveNode, Nodefield, FullCognitiveAgent,
    global_tick
)
from atlan_kernel_config import KernelConfig


class TestCognitiveNode:
    """Test basic node functionality"""
    
    def test_node_creation(self):
        node = CognitiveNode(0, (0, 0, 0), "test", "TestSymbol")
        assert node.node_id == 0
        assert node.position == (0, 0, 0)
        assert node.context_space == "test"
        assert node.symbolic_anchor == "TestSymbol"
        assert node.resonance_energy == 0.0
        
    def test_distance_calculation(self):
        node1 = CognitiveNode(0, (0, 0, 0), "test")
        node2 = CognitiveNode(1, (1, 1, 1), "test")
        distance = node1.distance_to(node2)
        assert abs(distance - np.sqrt(3)) < 0.001


class TestNodefield:
    """Test nodefield operations"""
    
    def test_add_node(self):
        field = Nodefield()
        node = field.add_node((0, 0, 0), "test", "Symbol")
        assert (0, 0, 0) in field.nodes
        assert field.nodes[(0, 0, 0)].symbolic_anchor == "Symbol"
        
    def test_energy_propagation(self):
        global global_tick
        global_tick = 0
        
        field = Nodefield()
        field.add_node((0, 0, 0), "test", "Source")
        field.add_node((1, 0, 0), "test", "Target")
        
        global_tick += 1
        logs = field.propagate_resonance((0, 0, 0), 5.0, 1.0)
        
        # Check source received energy
        assert field.nodes[(0, 0, 0)].resonance_energy > 0
        # Check target received transferred energy
        assert field.nodes[(1, 0, 0)].resonance_energy > 0
        # Check decay was applied
        assert field.nodes[(0, 0, 0)].resonance_energy < 5.0
        
    def test_memory_chain(self):
        field = Nodefield()
        field.add_node((0, 0, 0), "test", "Node1")
        
        global global_tick
        global_tick = 1
        field.propagate_resonance((0, 0, 0), 5.0, 1.0)
        
        assert len(field.memory_chain) > 0
        assert field.memory_chain[0][2] == "Node1"  # target symbol

    def test_default_config_usage(self):
        """propagate_resonance should use CONFIG.dampening when not supplied."""
        field = Nodefield()
        field.add_node((0, 0, 0), "test", "Src")
        field.add_node((1, 0, 0), "test", "Tgt")

        global global_tick
        global_tick = 1
        # Call without dampening arg
        field.propagate_resonance((0, 0, 0), 2.0, 1.0)

        # Call with explicit arg equal to default config; energies should differ by <1e-6
        before = field.nodes[(1, 0, 0)].resonance_energy
        # reset energies
        for n in field.nodes.values():
            n.resonance_energy = 0.0
        global_tick += 1
        field.propagate_resonance((0, 0, 0), 2.0, 1.0, KernelConfig().dampening)
        after = field.nodes[(1, 0, 0)].resonance_energy

        assert abs(before - after) < 1e-6


class TestAbstraction:
    """Test abstraction formation"""
    
    def test_abstraction_threshold(self):
        brain = FullCognitiveAgent()
        brain.add_node((0, 0, 0), "test", "Concept1")
        brain.add_node((1, 0, 0), "test", "Concept2")
        
        # Set different energy levels
        brain.nodes[(0, 0, 0)].resonance_energy = 1.5
        brain.nodes[(1, 0, 0)].resonance_energy = 0.5
        
        abstractions, _ = brain.derive_abstractions(1.0)
        
        assert "Abstract_Concept1" in abstractions
        assert "Abstract_Concept2" not in abstractions


class TestSemanticLinks:
    """Test semantic relationships"""
    
    def test_semantic_linking(self):
        brain = FullCognitiveAgent()
        brain.add_node((0, 0, 0), "test", "Parent")
        brain.add_node((1, 0, 0), "test", "Child")
        
        brain.add_semantic_link("Parent", "Child")
        
        assert ("Parent", "Child") in brain.semantic_links


class TestPrediction:
    """Test predictive capabilities"""
    
    def test_prediction_model(self):
        brain = FullCognitiveAgent()
        
        # Create a simple memory chain
        brain.memory_chain = [
            (1, None, "A", 1.0),
            (2, "A", "B", 0.5),
            (3, "B", "C", 0.5),
            (4, "A", "B", 0.5),  # A->B happens twice
        ]
        
        brain.build_prediction_model()
        
        # Should predict B after A with high probability
        assert "A" in brain.prediction_model
        predictions = brain.prediction_model["A"]
        assert any(sym == "B" for sym, _ in predictions)


class TestReinforcement:
    """Test reward-based learning"""
    
    def test_reward_application(self):
        brain = FullCognitiveAgent()
        brain.add_node((0, 0, 0), "test", "Concept")
        
        initial_energy = brain.nodes[(0, 0, 0)].resonance_energy
        brain.apply_rewards({"Concept": 1.0})
        final_energy = brain.nodes[(0, 0, 0)].resonance_energy
        
        assert final_energy == initial_energy + 1.0
        assert len(brain.reinforcement_log) == 1


class TestConflictResolution:
    """Test belief conflict handling"""
    
    def test_conflict_detection(self):
        brain = FullCognitiveAgent()
        brain.add_node((0, 0, 0), "test", "Belief1")
        brain.add_node((1, 0, 0), "test", "Belief2")
        
        # Set similar energy levels
        brain.nodes[(0, 0, 0)].resonance_energy = 1.0
        brain.nodes[(1, 0, 0)].resonance_energy = 1.1
        
        brain.detect_conflicts([("Belief1", "Belief2")], threshold=0.2)
        
        assert len(brain.conflict_log) == 1
        
        brain.resolve_conflicts()
        # Should favor Belief2 due to higher energy


class TestIdentity:
    """Test identity formation"""
    
    def test_identity_core(self):
        brain = FullCognitiveAgent()
        brain.add_node((0, 0, 0), "test", "CoreBelief")
        brain.add_node((1, 0, 0), "test", "WeakBelief")
        
        brain.nodes[(0, 0, 0)].resonance_energy = 2.0
        brain.nodes[(1, 0, 0)].resonance_energy = 0.5
        
        brain.form_identity_model(core_threshold=1.0)
        
        assert "CoreBelief" in brain.identity_core
        assert "WeakBelief" not in brain.identity_core


class TestLifeCycle:
    """Test autonomous evolution"""
    
    def test_life_cycle_execution(self):
        brain = FullCognitiveAgent()
        
        # Seed some initial knowledge
        positions = [(i, 0, 0) for i in range(5)]
        for i, pos in enumerate(positions):
            brain.add_node(pos, "test", f"Concept{i}")
            brain.nodes[pos].resonance_energy = 0.5 * i
            
        # Run a single life cycle
        logs = brain.run_life_cycle(cycles=1)
        
        # Should have executed without errors
        assert len(logs) > 0
        # Should have formed value hierarchy
        assert hasattr(brain, 'intention_map')
        # Should have identity model
        assert hasattr(brain, 'identity_core')


class TestIntegration:
    """Integration tests for full system"""
    
    def test_full_cognitive_flow(self):
        """Test complete cognitive pipeline"""
        brain = FullCognitiveAgent()
        
        # 1. Create knowledge
        brain.add_node((0, 0, 0), "bio", "Plant")
        brain.add_node((1, 0, 0), "bio", "Tree")
        brain.add_semantic_link("Tree", "Plant")
        
        # 2. Learn
        global global_tick
        global_tick = 1
        brain.propagate_resonance((0, 0, 0), 5.0, 0.75)
        
        # 3. Consolidate
        brain.replay_memory_chain()
        
        # 4. Abstract
        abstractions, _ = brain.derive_abstractions(0.5)
        
        # 5. Predict
        brain.build_prediction_model()
        
        # 6. Self-monitor
        brain.assess_belief_stability(0.5)
        
        # 7. Form identity
        brain.form_identity_model(0.5)
        
        # All systems should be populated
        assert len(brain.memory_chain) > 0
        assert len(abstractions) > 0
        assert hasattr(brain, 'stable_beliefs')
        assert len(brain.identity_core) > 0


def test_importance_weights():
    """Test different importance weight effects"""
    brain = FullCognitiveAgent()
    
    # Create chain of nodes
    for i in range(5):
        brain.add_node((i, 0, 0), "test", f"Node{i}")
        
    global global_tick
    
    # Test with high importance
    global_tick = 1
    brain.propagate_resonance((0, 0, 0), 5.0, 1.0)
    high_energy = brain.nodes[(1, 0, 0)].resonance_energy
    
    # Reset
    for node in brain.nodes.values():
        node.resonance_energy = 0.0
        
    # Test with low importance
    global_tick = 2
    brain.propagate_resonance((0, 0, 0), 5.0, 0.1)
    low_energy = brain.nodes[(1, 0, 0)].resonance_energy
    
    assert high_energy > low_energy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])