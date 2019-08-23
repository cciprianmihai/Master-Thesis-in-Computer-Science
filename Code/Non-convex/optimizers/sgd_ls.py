import torch
from torch.optim.optimizer import Optimizer, required
from functools import reduce
from math import isinf
import numpy as np


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        line_search (str, optional): line search method for learning rate (default: None)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, line_search=None, tolerance_grad=1e-5, tolerance_change=1e-9, a_1=0.0, a_2=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        line_search=line_search, tolerance_grad=tolerance_grad,
                        tolerance_change=tolerance_change, a_1=a_1, a_2=a_2)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self.param_state = dict(momentum_buffer=None)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(),
                                       self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_data(self):
        views = []
        for p in self._params:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(step_size,
                        update[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        line_search = group['line_search']
        a_1 = group['a_1']
        a_2 = group['a_2']

        # Apply weight decay update
        if weight_decay != 0:
            for p in group['params']:
                d_p = p.grad.data
                d_p.add_(weight_decay, p.data)

        flat_grad = self._gather_flat_grad()

        # Apply momentum update
        if momentum != 0:
            if self.param_state['momentum_buffer'] is None:
                buf = self.param_state['momentum_buffer'] = torch.zeros_like(flat_grad)
                buf.mul_(momentum).add_(flat_grad)
            else:
                buf = self.param_state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, flat_grad)
            if nesterov:
                flat_grad.add_(momentum, buf)
            else:
                flat_grad = buf

        # Get the direction of the update
        d = flat_grad.neg()

        # Get the step size of the update
        if line_search is None:
            t = group['lr']
        else:
            if line_search == 'weak_wolfe':
                t = self._line_search_weak_wolfe(closure, d, a_1, a_2)
            elif line_search == 'goldstein':
                t = self._line_search_goldstein(closure, d, a_1, a_2)
            elif line_search == 'backtracking':
                t = self._line_search_backtracking(closure, d, a_2)
            elif line_search == 'blind':
                t = self._line_search_blind(closure, d)
        self._add_grad(t, d)

        return loss, t

    def _save_model_parameters(self):
        original_param_data_list = []
        for p in self._params:
            param_data = p.data.new(p.size())
            param_data.copy_(p.data)
            original_param_data_list.append(param_data)
        return original_param_data_list

    def _set_param(self, param_data_list):
        for i in range(len(param_data_list)):
            self._params[i].data.copy_(param_data_list[i])

    def _update_model_parameters(self, alpha, d):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(
                p.data + alpha * d[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()

    def _directional_derivative(self, d):
        deriv = 0.0
        offset = 0
        for p in self._params:
            numel = p.numel()
            deriv += torch.sum(
                p.grad.data * d[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()
        return deriv

    def _line_search_backtracking(self, closure, d, alpha_k):
        """
        Back tracking line search method with the following preconditions:
            1. 0 < rho < 0.5
            2. 0 < w < 1
        """
        rho = 1e-4
        w = 0.5
        # Save initial model parameters
        initial_model_parameters = self._save_model_parameters()
        # Compute initial loss
        f_0 = closure().item()
        # Compute the directional derivative
        f_0_prime = self._directional_derivative(d)
        # While conditions are True
        while True:
            # Update parameters with the value of alpha_k
            self._update_model_parameters(alpha_k, d)
            # Calculate the loss of the new model
            f_k = closure().item()
            # Restore the initial model
            self._set_param(initial_model_parameters)
            # If conditions are met, we stop
            if f_k <= f_0 + rho * alpha_k * f_0_prime:
                break
            # Else we update alpha_k for a new iteration
            else:
                alpha_k *= w

        # sum = 0
        # final_model_parameters = self._save_model_parameters()
        # for i in range(len(final_model_parameters)):
        #     s = final_model_parameters[i].eq(initial_model_parameters[i]).sum()
        #     print("Intermediar s:", s)
        #     sum += s
        # print("Sum:", sum)
        # print("Eq first lists:", final_model_parameters[0].eq(initial_model_parameters[0]))
        # print("Len first lists:", len(final_model_parameters[0].eq(initial_model_parameters[0])))

        # Return alpha_k - step size
        return alpha_k

    def _line_search_goldstein(self, closure, d, a_1, a_2):
        """
        Goldstein line search method with the following preconditions:
            1. 0 < rho < 0.5
            2. t > 1
        """
        rho = 1e-4
        t = 2.0
        # Save initial model parameters
        initial_model_parameters = self._save_model_parameters()
        # Compute initial loss
        f_0 = closure().item()
        # Compute the directional derivative
        f_0_prime = self._directional_derivative(d)
        # Set initial alpha_k
        alpha_k = min(1e4, (a_1 + a_2) / 2.0)
        # While conditions are True
        while True:
            # Update parameters with the value of alpha_k
            self._update_model_parameters(alpha_k, d)
            # Calculate the loss of the new model
            f_k = closure().item()
            # Restore the initial model
            self._set_param(initial_model_parameters)
            # If conditions are met, we stop, else we update alpha_k for a new iteration
            if f_k <= f_0 + rho * alpha_k * f_0_prime:
                if f_k >= f_0 + (1 - rho) * alpha_k * f_0_prime:
                    break
                else:
                    a_1 = alpha_k
                    alpha_k = t * alpha_k if isinf(a_2) else (a_1 + a_2) / 2.0
            else:
                a_2 = alpha_k
                alpha_k = (a_1 + a_2) / 2.0
            if torch.sum(torch.abs(alpha_k * d)) < self.param_groups[0]['tolerance_grad']:
                break
            if abs(a_2 - a_1) < 1e-6:
                break
        # Return alpha_k - step size
        return alpha_k

    def _line_search_weak_wolfe(self, closure, d, a_1, a_2):
        """
        Weak Wolfe line search method with the following preconditions:
            1. 0 < rho < 0.5
            2. rho < sigma < 1
        """
        rho = 1e-4
        sigma = 0.9
        # Save initial model parameters
        initial_model_parameters = self._save_model_parameters()
        # Compute initial loss
        f_0 = closure().item()
        # Compute the directional derivative
        f_0_prime = self._directional_derivative(d)
        # Set initial alpha_k
        alpha_k = min(1e4, (a_1 + a_2) / 2.0)
        # While conditions are True
        while True:
            # Update parameters with the value of alpha_k
            self._update_model_parameters(alpha_k, d)
            # Calculate the loss of the new model
            f_k = closure().item()
            # Compute the directional derivative
            f_k_prime = self._directional_derivative(d)
            # Restore the initial model
            self._set_param(initial_model_parameters)
            # If conditions are met, we stop, else we update alpha_k for a new iteration
            if f_k <= f_0 + rho * alpha_k * f_0_prime:
                if f_k_prime >= sigma * f_0_prime:
                    break
                else:
                    alpha_hat = alpha_k + (alpha_k - a_1) * f_k_prime / (f_0_prime - f_k_prime)
                    a_1 = alpha_k
                    f_0 = f_k
                    f_0_prime = f_k_prime
                    alpha_k = alpha_hat
            else:
                alpha_hat = a_1 + 0.5 * (alpha_k - a_1) / (1 + (f_0 - f_k) / ((alpha_k - a_1) * f_0_prime))
                a_2 = alpha_k
                alpha_k = alpha_hat
            # We check, also, the tolerance grad
            if torch.sum(torch.abs(alpha_k * d)) < self.param_groups[0]['tolerance_grad']:
                break
            if abs(a_2 - a_1) < 1e-6:
                break
        # Return alpha_k - step size
        return alpha_k

    def _line_search_blind(self, closure, d, base_lr=0.0001):
        # Start value for alpha_k
        alpha_k = base_lr
        # Multiplication factor
        w = 1.5
        # Save initial model parameters
        initial_model_parameters = self._save_model_parameters()
        # Update parameters with the value of alpha_k
        self._update_model_parameters(alpha_k, d)
        # Compute loss after a basic SGD step
        f_0 = closure().item()
        # Restore the initial model
        self._set_param(initial_model_parameters)

        while True:
            # Attempt to increase step size, compute loss at that point
            alpha_k *= w
            # Update parameters with the value of alpha_k
            self._update_model_parameters(alpha_k, d)
            # Compute loss with the new parameters
            f_k = closure().item()
            # Restore the initial model
            self._set_param(initial_model_parameters)
            # If conditions are met, we stop
            if f_k >= f_0:
                alpha_k /= w
                break
            else:
                f_0 = f_k
        # Return alpha_k - step size
        return alpha_k

    def _grad_norm(self):
        flat_grad = self._gather_flat_grad()
        return flat_grad.norm()
