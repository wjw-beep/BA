from .io import read_data


def load_config(path):
    config = read_data(path)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {
        "protocol": ("retention_probability", "probes", "master_seed", "namespace"),
        "inference": ("influence_probability", "influence_cutoff", "smoothing_grid", "regularization_C", "max_iter", "threshold_grid", "confidence_alpha"),
        "ranking": ("lambda_penalty", "evidence_budget", "unknown_pair_penalty"),
        "oracle": ("max_exact_origins", "monte_carlo_samples"),
    }
    for section, names in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
        for name in names:
            if name not in config[section]:
                raise ValueError(f"Missing configuration key: {section}.{name}")
    p = config["protocol"]["retention_probability"]
    p0 = config["inference"]["influence_probability"]
    if not 0 < p < 1 or not 0 < p0 < 1:
        raise ValueError("Retention probabilities must be inside (0,1)")
    if not 0 < config["inference"]["influence_cutoff"] < 1:
        raise ValueError("influence_cutoff must be inside (0,1)")
    if config["inference"]["regularization_C"] <= 0 or config["inference"]["max_iter"] < 1:
        raise ValueError("Model regularization and iteration budget must be positive")
    if config["ranking"]["lambda_penalty"] < 0:
        raise ValueError("Redundancy penalty must be nonnegative")
    if config["protocol"]["probes"] <= 0:
        raise ValueError("Probe count must be positive")
    if not config["inference"]["smoothing_grid"] or any(h <= 0 for h in config["inference"]["smoothing_grid"]):
        raise ValueError("Smoothing grid must contain positive values")
    if not config["inference"]["threshold_grid"] or any(not 0 <= t <= 1 for t in config["inference"]["threshold_grid"]):
        raise ValueError("Threshold grid must contain numbers in [0,1]")
    if not 0 < config["inference"]["confidence_alpha"] < 1:
        raise ValueError("confidence_alpha must be inside (0,1)")
    if config["ranking"]["evidence_budget"] <= 0:
        raise ValueError("Evidence budget must be positive")
    if not 0 <= config["ranking"]["unknown_pair_penalty"] <= 1:
        raise ValueError("unknown_pair_penalty must be in [0,1]")
    if config["oracle"]["max_exact_origins"] < 1 or config["oracle"]["monte_carlo_samples"] < 1:
        raise ValueError("Oracle limits must be positive")
    return config
