#!/usr/bin/env python3

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import common, fit_model, inv_softplus, log_pdf_normal


@dataclass(frozen=True)
class LogDailyReturnParams:
    var_0: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor


class LogDailyReturnModel(nn.Module):
    def __init__(
        self,
        mu: float,
        var_0: float,
        alpha: float,
        beta: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        if var_0 <= 0:
            raise ValueError(f"Invalid parameters: var_0 must be positive, but got var_0={var_0}")
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Invalid parameters: alpha and beta must be positive, but got alpha={alpha}, beta={beta}")
        if alpha + beta >= 1:
            raise ValueError(f"Invalid parameters: alpha + beta must be less than 1, but got alpha={alpha}, beta={beta}")

        self.mu = torch.tensor(mu, dtype=dtype)
        self.unconstrained_var_0 = nn.Parameter(inv_softplus(torch.tensor(var_0, dtype=dtype)))
        gamma = 1 - alpha - beta
        self.unconstrained_alpha = nn.Parameter(torch.log(torch.tensor(alpha / gamma, dtype=dtype)))
        self.unconstrained_beta = nn.Parameter(torch.log(torch.tensor(beta / gamma, dtype=dtype)))

        self.dtype = dtype

    def _params(self) -> LogDailyReturnParams:
        var_0 = F.softplus(self.unconstrained_var_0)
        zero = self.unconstrained_var_0.new_zeros(())
        alpha, beta, gamma = F.softmax(torch.stack([self.unconstrained_alpha, self.unconstrained_beta, zero]), dim=0)

        return LogDailyReturnParams(
            var_0=var_0,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def print_params(self) -> None:
        params = self._params()
        print(f"Fixed parameters: mu={self.mu.item():.6f}")
        print(f"Current parameters: var_0={params.var_0.item():.6f}, alpha={params.alpha.item():.6f}, beta={params.beta.item():.6f}, gamma={params.gamma.item():.6f}")

    def _negative_log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        params = self._params()
        omega = params.var_0 * params.gamma
        log_likelihood = params.var_0.new_zeros(())
        var_i = params.var_0
        for i in range(x.shape[0]):
            log_likelihood += log_pdf_normal(x[i], self.mu, var_i)
            var_i = omega + params.alpha * (x[i] - self.mu).square() + params.beta * var_i
        return -log_likelihood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            raise ValueError(f"Invalid input: expected dtype={self.dtype}, but got dtype={x.dtype}")

        return self._negative_log_likelihood(x)

    def sample_log_annual_return(self, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            params = self._params()
            omega = params.var_0 * params.gamma
            samples = params.var_0.new_zeros((num_samples))
            var_i = params.var_0.repeat(num_samples)
            for _ in range(252):
                x_i = torch.normal(self.mu, torch.sqrt(var_i))
                samples += x_i
                var_i = omega + params.alpha * (x_i - self.mu).square() + params.beta * var_i
            return samples


def main() -> None:
    series_of_daily_return, _, args = common()

    # Compute series of log daily return
    series_of_log_daily_return: npt.NDArray[np.float64] = np.log(series_of_daily_return)

    # Initialize model
    mu: np.float64 = series_of_log_daily_return.mean()
    var_0: np.float64 = series_of_log_daily_return.var(ddof=1)
    alpha = 0.1
    beta = 0.85
    model = LogDailyReturnModel(mu, var_0, alpha, beta, dtype=torch.float64)
    model.print_params()

    # Fit model
    data = torch.from_numpy(series_of_log_daily_return).to(torch.float64)
    fit_model(model, data, lr=0.5, num_iterations=200)
    model.print_params()

    # Sample log annual return
    samples_of_log_annual_return = model.sample_log_annual_return(100000).reshape((1000, 100)).detach().numpy()

    # Compute 90% confidence interval of annual volatility
    lower_bound_of_annual_volatility = np.percentile(np.sqrt(samples_of_log_annual_return.var(ddof=1, axis=1)), 5)
    upper_bound_of_annual_volatility = np.percentile(np.sqrt(samples_of_log_annual_return.var(ddof=1, axis=1)), 95)
    print(f"90% confidence interval of annual volatility: [{lower_bound_of_annual_volatility:.2f}, {upper_bound_of_annual_volatility:.2f}]")

    # Compute 80% confidence interval of mean of log annual return
    lower_bound_of_mean_of_log_annual_return = np.percentile(samples_of_log_annual_return.mean(axis=1), 10)
    upper_bound_of_mean_of_log_annual_return = np.percentile(samples_of_log_annual_return.mean(axis=1), 90)
    print(f"80% confidence interval of mean of log annual return: [{lower_bound_of_mean_of_log_annual_return:.2f}, {upper_bound_of_mean_of_log_annual_return:.2f}]")

    # Compute 50% confidence interval of mean of log annual return
    sublower_bound_of_mean_of_log_annual_return = np.percentile(samples_of_log_annual_return.mean(axis=1), 25)
    subupper_bound_of_mean_of_log_annual_return = np.percentile(samples_of_log_annual_return.mean(axis=1), 75)
    print(f"50% confidence interval of mean of log annual return: [{sublower_bound_of_mean_of_log_annual_return:.2f}, {subupper_bound_of_mean_of_log_annual_return:.2f}]")

    # Compute near-worst-case annual return
    def cdf_of_near_worst_case_annual_return(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.percentile((samples_of_log_annual_return[:, :, None] <= np.log(x)[None, None, :]).mean(axis=1), 90, axis=0)

    # Compute subnear-worst-case annual return
    def cdf_of_subnear_worst_case_annual_return(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.percentile((samples_of_log_annual_return[:, :, None] <= np.log(x)[None, None, :]).mean(axis=1), 75, axis=0)

    # Compute series of annual return
    series_of_annual_return: npt.NDArray[np.float64] = series_of_daily_return[series_of_daily_return.shape[0] % 252:].reshape((-1, 252)).prod(axis=1)

    # Plot annual return
    x = np.linspace(1e-8, 3, 1000)
    y1 = cdf_of_near_worst_case_annual_return(x)
    plt.plot(x, y1, label="Near-worst-case Theoretical")
    y2 = cdf_of_subnear_worst_case_annual_return(x)
    plt.plot(x, y2, label="Subnear-worst-case Theoretical")
    plt.hist(series_of_annual_return, bins=series_of_annual_return.shape[0], density=True, cumulative=True, alpha=0.5, label="Empirical")
    plt.xlabel("Annual return")
    plt.xlim(0, 3)
    plt.ylabel("Cumulative probability")
    plt.ylim(0, 1)
    plt.legend()
    if args.path_to_plot is not None:
        plt.savefig(args.path_to_plot)
    else:
        plt.show()


if __name__ == "__main__":
    main()
