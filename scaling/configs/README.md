# Configuration Guidelines

Store YAML configuration files here for training, evaluation, and sweep definitions. Prefer snake_case keys and explicit defaults to simplify downstream tooling.

```yaml
# example: scaling/configs/base_experiment.yaml
experiment_name: placeholder_experiment
model_size: small
optimizer:
  name: adamw
  learning_rate: 0.0005
```

> TODO: Document required fields, validation checks, and environment variable support.
