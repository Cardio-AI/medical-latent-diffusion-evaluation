"""Load compatible first-stage checkpoints without silently missing weights."""


def load_stage1_weights(model, checkpoint):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("First-stage checkpoint must contain a state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    wanted = set(model.state_dict())
    missing = sorted(wanted - set(state))
    if missing:
        raise ValueError(f"First-stage checkpoint is missing {len(missing)} required keys: {missing[:5]}")
    extra = sorted(set(state) - wanted)
    if extra:
        print(f"Ignoring {len(extra)} additional first-stage keys: {extra[:5]}")
    model.load_state_dict({key: state[key] for key in wanted}, strict=True)
