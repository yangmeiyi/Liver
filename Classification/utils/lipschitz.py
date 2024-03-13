from torch import autograd
import torch

# Class for lipschitz constraint(start)
class LipschitzConstraint:

    # Initialization(start)
    def __init__(self, discriminator):
        self._discriminator = discriminator
    # Initialization(end)

    # Prepare for discriminator(start)
    def prepare_discriminator(self):

        raise NotImplementedError()
    # Prepare for discriminator(end)

    # Calculate penalty(start)
    def calculate_loss_penalty(self, real_var, fake_var):

        raise NotImplementedError()
    # Calculate penalty(end)
# Class for lipschitz constraint(end)

# Class for gradient penalty(start)
class GradientPenalty(LipschitzConstraint):

    # Initialization(start)
    def __init__(self, discriminator, coefficient=10):

        super().__init__(discriminator)
        self._coefficient = coefficient
    # Initialization(end)

    # Calculate gradient penalty(start)
    def cal_gradient(self, real, fake):

        assert real.size(0) == fake.size(0)
        batch = real.size(0)
        alpha = torch.rand(batch, 1, 1, 1)
        alpha = alpha.expand_as(real)
        alpha = alpha.type_as(real)
        interp_data = alpha * real + ((1 - alpha) * fake)
        interp_data = autograd.Variable(interp_data, requires_grad=True)
        disc_out, _, class_out = self._discriminator(interp_data)
        grad_outputs = torch.ones(disc_out.size()).type_as(disc_out.data)
        gradients = autograd.grad(
            outputs=disc_out,
            inputs=interp_data,
            grad_outputs=grad_outputs,
            create_graph=True,
            only_inputs=True)[0]
        gradients = gradients.view(batch, -1)
        gradient_penalty = self._coefficient * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()
        return gradient_penalty
    # Calculate gradient penalty(end)
# Class for gradient penalty(end)