from dataclasses import dataclass

@dataclass
class KernelConfig:
    """Central configuration for Atlan Brain Kernel constants.

    Keeping these values in one place makes it easy to run parameter sweeps
    and ensures that experiments are reproducible without touching the core
    algorithmic code.
    """
    # Energy dynamics ----------------------------------------------------
    default_decay_factor: float = 0.1  # Node energy decay per tick
    activation_threshold: float = 1.0  # Minimum energy to start propagation
    dampening: float = 0.5            # Energy dampening during propagation
    epsilon: float = 0.01             # Small constant to avoid div-by-zero

    # Performance -------------------------------------------------------
    vectorized: bool = False          # Use NumPy vectorized propagation

    # Persistence -------------------------------------------------------
    db_path: str = "memory_chain.sqlite"  # SQLite DB for memory storage

    # Logging -------------------------------------------------------------
    log_level: str = "INFO"           # Default log level 