# Hyperparameter Sweeps

Place sweep definitions and scheduling scripts here. Keep YAML sweep files minimal, relying on anchors and references to avoid duplication.

```yaml
# example: scaling/sweeps/learning_rate_sweep.yaml
include: ../configs/base_experiment.yaml
parameters:
  optimizer.learning_rate:
    values: [0.0001, 0.0003, 0.001]
```

> TODO: Document supported sweep runners (e.g., WandB, Ray Tune) and expected CLI options.
