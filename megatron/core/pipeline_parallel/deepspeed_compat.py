"""Narrow compatibility fixes for DeepSpeed pipeline execution."""


def configure_pipeline_gradient_scaling(engine):
    """Prevent repeated GAS scaling on non-final pipeline stages.

    The final stage divides the loss gradient by GAS.  That already-scaled
    activation gradient is sent to the preceding stage, so DeepSpeed's output
    backward hook must not divide it again on non-final stages.
    """
    if engine.is_last_stage():
        return
    if not hasattr(engine, "_scale_wrt_gas"):
        raise RuntimeError(
            "DeepSpeed PipelineEngine does not expose _scale_wrt_gas; "
            "cannot prevent repeated GAS scaling safely"
        )
    engine._scale_wrt_gas = False
