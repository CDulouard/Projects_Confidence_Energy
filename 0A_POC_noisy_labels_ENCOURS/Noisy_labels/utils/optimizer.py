import math
from typing import Callable, Optional

import torch
import torch.optim
        
import math
import torch
from torch.optim.optimizer import Optimizer


from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

__all__ = ("MADGRAD",)


class MADGRAD(torch.optim.Optimizer):
    r"""Implements MADGRAD algorithm.

    It has been proposed in `Adaptivity without Compromise: A Momentumized,
    Adaptive, Dual Averaged Gradient Method for Stochastic Optimization`__

    Arguments:
        params (iterable):
            Iterable of parameters to optimize
            or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation
            to improve numerical stability. (default: 1e-6).

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.MAGRAD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2101.11075

    Note:
        Reference code: https://github.com/facebookresearch/madgrad
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay, k=0
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                state["s"] = torch.zeros_like(p.data).detach()
                if momentum != 0:
                    state["x0"] = torch.clone(p.data).detach()

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            k = group["k"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError(
                        "momentum != 0 is not compatible with "
                        "sparse gradients"
                    )

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not "
                            "compatible with sparse gradients"
                        )

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = (
                        grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    )
                    x0_masked_vals = p_masked._values().addcdiv(
                        s_masked._values(), rms_masked_vals, value=1
                    )

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = (
                        grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)
                    )

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(), rms_masked_vals, value=-1
                    )
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

            group["k"] = group["k"] + 1
        return loss
        
        

class FastAdaBelief(Optimizer):
    """
    Implements the FastAdaBelief optimization algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): L2 regularization coefficient (default: 0).
        amsgrad (bool, optional): Whether to use the AMSGrad variant of AdaBelief (default: False).
        weight_decouple (bool, optional): Whether to decouple the weight decay from the gradient-based optimization step (default: False).
        fixed_decay (bool, optional): Whether to use a fixed decay factor for weight decay (default: False).
        rectify (bool, optional): Whether to use rectification in the optimization step (default: False).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True, weight_decouple=True, fixed_decay=True, rectify=True):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if weight_decay < 0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(FastAdaBelief, self).__init__(params, defaults)

        self._weight_decouple = weight_decouple
        self._rectify = rectify
        self._fixed_decay = fixed_decay

    def __setstate__(self, state):
        super(FastAdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value evaluated by the closure.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if self._weight_decouple:
                    if not self._fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # ici STEP 8
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self._rectify:
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (1.0 - beta2 ** state['step'])

                    if state['rho_t'] > 4:
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t)
                        rt = math.sqrt(rt)

                        step_size = rt * group['lr'] / bias_correction1

                        p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    else:
                        p.data.add_(exp_avg, alpha=-group['lr'])

        return loss
