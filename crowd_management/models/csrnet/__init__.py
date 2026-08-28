"""CSRNet crowd density estimation module."""

from crowd_management.models.csrnet.architecture import CSRNet
from crowd_management.models.csrnet.infer import CSRNetInference
from crowd_management.models.csrnet.weights import build_csrnet_model

__all__ = ["CSRNet", "CSRNetInference", "build_csrnet_model"]
