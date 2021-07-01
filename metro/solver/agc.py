import copy
from typing import Optional, Type

import torch
from torch.cuda.amp.grad_scaler import OptState

from detectron2.solver.build import _GradientClipper


def add_agc(cfg, optimizer_type):
    assert issubclass(optimizer_type, torch.optim.Optimizer), optimizer_type

    grad_clipper = _create_adaptive_gradient_clipper(cfg.SOLVER.AGC)
    OptimizerWithGradientClip = _generate_optimizer_class_with_agc(optimizer_type, per_param_clipper=grad_clipper)
    return OptimizerWithGradientClip


def _create_adaptive_gradient_clipper(cfg):
    cfg = copy.deepcopy(cfg)

    def adaptive_gradient_clip(p: torch.Tensor):
        param_norm = torch.max(unitwise_norm(p.detach()), torch.tensor(cfg.EPS).to(p.device))
        grad_norm = unitwise_norm(p.grad.detach())
        max_norm = param_norm * cfg.CLIP_VALUE

        trigger = grad_norm > max_norm
        clipped_grad = p.grad * (max_norm / torch.max(grad_norm, torch.tensor(1e-6).to(grad_norm.device)))
        p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

    return adaptive_gradient_clip


def _generate_optimizer_class_with_agc(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    def optimizer_wgc_step(self, closure=None, grad_scaler=None):
        # Unscale gradient for FP16 training.
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        if grad_scaler:
            optimizer_state = grad_scaler._per_optimizer_states[id(self)]
            assert optimizer_state["stage"] == OptState.READY
            grad_scaler.unscale_(self)

        for group in self.param_groups:
            if group['apply_agc']:
                for p in group["params"]:
                    per_param_clipper(p)

        if grad_scaler:
            # torch 1.8
            # return grad_scaler._maybe_opt_step(super(type(self), self), optimizer_state)
            # Skip optimizer.step() if INF found in gradients.
            retval = None
            if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                # Actually update parameters.
                retval = super(type(self), self).step()
            return retval
        else:
            return super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithAGC",
        (optimizer, ),
        {
            "step": optimizer_wgc_step,
            "_step_supports_amp_scaling": True
        },
    )
    return OptimizerWithGradientClip


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim)**0.5
