import torch
import numpy as np
from .assemble import (
    assemble_single_boundary_matrix,
    assemble_double_boundary_matrix,
    assemble_identity_matrix,
    assemble_double_potential_matrix,
    assemble_single_potential_matrix,
)

import torch
import warnings


class BiCGSTAB:
    """
    modified from https://gist.github.com/bridgesign/f421f69ad4a3858430e5e235bccde8c6
    This is a pytorch implementation of BiCGSTAB solver.
    """

    def __init__(self, A, preconditioner=None, device="cuda"):
        self.A = A
        self.preconditioner = preconditioner
        self.device = device

    def matvec(self, x):
        if self.preconditioner is None:
            return self.A @ x
        else:
            return self.preconditioner(self.A @ x)

    def init_params(self, b, x=None, max_iter=None, tol=1e-10, atol=1e-16):
        """
        b: The R.H.S of the system. 1-D tensor
        max_iter: Number of steps of calculation
        tol: Tolerance such that if ||r||^2 < tol * ||b||^2 then converged
        atol:  Tolernace such that if ||r||^2 < atol then converged
        """
        self.b = b
        self.x = (
            torch.zeros(b.shape[0], device=self.device, dtype=b.dtype)
            if x is None
            else x
        )
        self.residual_tol = tol * torch.vdot(b, b).item()
        self.atol = torch.tensor(atol, device=self.device, dtype=b.dtype)
        self.max_iter = b.shape[0] if max_iter is None else max_iter
        self.status, self.r = self.check_convergence(self.x)
        self.rho = torch.tensor(1, device=self.device, dtype=b.dtype)
        self.alpha = torch.tensor(1, device=self.device, dtype=b.dtype)
        self.omega = torch.tensor(1, device=self.device, dtype=b.dtype)
        self.v = torch.zeros(b.shape[0], device=self.device, dtype=b.dtype)
        self.p = torch.zeros(b.shape[0], device=self.device, dtype=b.dtype)
        self.r_hat = self.r.clone().detach()

    def check_convergence(self, x):
        r = self.b - self.matvec(x)
        # print("r", r)
        rdotr = torch.vdot(r, r).real
        if rdotr < self.residual_tol or rdotr < self.atol:
            return True, r
        else:
            return False, r

    def step(self):
        # rho_i <- <r0, r^>
        rho = torch.dot(self.r, self.r_hat)
        # beta <- (rho_i/rho_{i-1}) x (alpha/omega_{i-1})
        beta = (rho / self.rho) * (self.alpha / self.omega)
        # rho_{i-1} <- rho_i  replaced self value
        self.rho = rho
        # p_i <- r_{i-1} + beta x (p_{i-1} - w_{i-1} v_{i-1}) replaced p self value
        self.p = self.r + beta * (self.p - self.omega * self.v)
        self.v = self.matvec(self.p)  # v_i <- Ap_i
        # alpha <- rho_i/<r^, v_i>
        self.alpha = self.rho / torch.dot(self.r_hat, self.v)
        # h_i <- x_{i-1} + alpha p_i
        self.h = self.x + self.alpha * self.p
        status, res = self.check_convergence(self.h)
        if status:
            self.x = self.h
            return True
        # s <- r_{i-1} - alpha v_i
        s = self.r - self.alpha * self.v
        t = self.matvec(s)  # t <- As
        # w_i <- <t, s>/<t, t>
        self.omega = torch.dot(t, s) / torch.dot(t, t)
        # x_i <- x_{i-1} + alpha p + w_i s
        self.x = self.h + self.omega * s
        status, res = self.check_convergence(self.x)
        if status:
            return True
        else:
            self.r = s - self.omega * t  # r_i <- s - w_i t
            return False

    def solve(self, b, x=None, max_iter=None, tol=1e-4, atol=1e-16, warning=False):
        """
        Method to find the solution.
        Returns the final answer of x
        """
        if self.preconditioner is not None:
            b = self.preconditioner(b)
        # print('Solving the system...')
        # print('b:', b)
        self.init_params(b, x, max_iter, tol, atol)
        if self.status:
            return self.x
        while self.max_iter:
            s = self.step()
            if s:
                return self.x
            if self.rho == 0:
                break
            self.max_iter -= 1
        if warning:
            warnings.warn("Convergence has failed :(")
        return self.x


class boundary_mesh:
    def __init__(self, vertices, triangles, wave_number):
        self.vertices = vertices
        self.triangles = triangles
        self.wave_number = wave_number

    def solve_dirichlet(self, neumann):
        partialG = assemble_double_boundary_matrix(
            self.vertices, self.triangles, self.wave_number
        )
        G = assemble_single_boundary_matrix(
            self.vertices, self.triangles, self.wave_number
        )
        identity_matrix = assemble_identity_matrix(self.vertices, self.triangles)

        A = identity_matrix - 2 * partialG
        b = -2 * (G @ neumann)
        return BiCGSTAB(A, device="cuda").solve(b)

    def solve_points(self, points, neumann, dirichlet):
        if type(points) == np.ndarray:
            points = torch.from_numpy(points).float().cuda()
        partialG = assemble_double_potential_matrix(
            self.vertices, self.triangles, points, self.wave_number
        )
        G = assemble_single_potential_matrix(
            self.vertices, self.triangles, points, self.wave_number
        )
        return partialG @ dirichlet + G @ neumann
