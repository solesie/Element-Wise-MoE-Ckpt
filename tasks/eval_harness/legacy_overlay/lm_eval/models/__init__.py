"""The paper evaluator supplies a Megatron model object directly."""


def get_model(model_name):
    raise RuntimeError(f"named lm-eval models are disabled in this overlay: {model_name}")
