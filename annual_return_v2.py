#!/usr/bin/env python3

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import common, fit_model, inv_softplus, log_doubled_pi


@dataclass(frozen=True)
class LogDailyReturnParams:
    mu: torch.Tensor
    var_0: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    residual: torch.Tensor
    omega: torch.Tensor


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

        residual = 1 - alpha - beta

        if var_0 <= 0:
            raise ValueError(f"Invalid parameters: var_0 must be positive, but got var_0={var_0}")
        if alpha <= 0:
            raise ValueError(f"Invalid parameters: alpha must be positive, but got alpha={alpha}")
        if beta <= 0:
            raise ValueError(f"Invalid parameters: beta must be positive, but got beta={beta}")
        if residual <= 0:
            raise ValueError(f"Invalid parameters: alpha + beta must be less than 1, but got alpha={alpha}, beta={beta}")

        self.unconstrained_mu = nn.Parameter(torch.tensor(mu * 252, dtype=dtype))
        self.unconstrained_var_0 = nn.Parameter(inv_softplus(torch.tensor(var_0 * 252, dtype=dtype)))
        self.unconstrained_alpha = nn.Parameter(torch.log(torch.tensor(alpha / residual, dtype=dtype)))
        self.unconstrained_beta = nn.Parameter(torch.log(torch.tensor(beta / residual, dtype=dtype)))

        self.dtype = dtype

    def _params(
        self,
        unconstrained_mu: torch.Tensor,
        unconstrained_var_0: torch.Tensor,
        unconstrained_alpha: torch.Tensor,
        unconstrained_beta: torch.Tensor,
    ) -> LogDailyReturnParams:
        mu = unconstrained_mu / 252
        var_0 = F.softplus(unconstrained_var_0) / 252
        zero: torch.Tensor = unconstrained_var_0.new_zeros(())
        alpha, beta, residual = F.softmax(torch.stack([unconstrained_alpha, unconstrained_beta, zero]), dim=0)
        omega = var_0 * residual
        return LogDailyReturnParams(
            mu=mu,
            var_0=var_0,
            alpha=alpha,
            beta=beta,
            residual=residual,
            omega=omega,
        )

    def print_params(self) -> None:
        params = self._params(
            self.unconstrained_mu,
            self.unconstrained_var_0,
            self.unconstrained_alpha,
            self.unconstrained_beta,
        )
        print(f"Current parameters: mu={params.mu.item():.6f}, var_0={params.var_0.item():.6f}, alpha={params.alpha.item():.6f}, beta={params.beta.item():.6f}, residual={params.residual.item():.6f}, omega={params.omega.item():.6f}")

    def _negative_log_likelihood(self, params: LogDailyReturnParams, x: torch.Tensor) -> torch.Tensor:
        squared_epsilon = (x - params.mu).square()
        subvar = params.omega + params.alpha * squared_epsilon
        var: list[torch.Tensor] = []
        var.append(params.var_0)
        for i in range(1, x.shape[0]):
            var.append(subvar[i-1] + params.beta * var[i-1])
        var: torch.Tensor = torch.stack(var)
        negative_log_likelihood = 0.5 * (log_doubled_pi + torch.log(var) + (squared_epsilon / var)).sum()
        return negative_log_likelihood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            raise ValueError(f"Invalid input: expected dtype={self.dtype}, but got dtype={x.dtype}")
        if x.ndim != 1:
            raise ValueError(f"Invalid input: expected 1-dimensional input, but got {x.ndim}-dimensional input")

        params = self._params(
            self.unconstrained_mu,
            self.unconstrained_var_0,
            self.unconstrained_alpha,
            self.unconstrained_beta,
        )
        return self._negative_log_likelihood(params, x)

    def _sample_log_annual_return(self, params: LogDailyReturnParams, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            log_annual_return = params.mu.new_zeros((num_samples,))
            var_i = params.var_0.repeat((num_samples,))
            for _ in range(252):
                x_i = torch.normal(params.mu, torch.sqrt(var_i))
                log_annual_return += x_i
                squared_epsilon_i = (x_i - params.mu).square()
                var_i = params.omega + params.alpha * squared_epsilon_i + params.beta * var_i
            return log_annual_return

    def _sample_mdd_days(self, params: LogDailyReturnParams, num_samples: int, num_days: int) -> torch.Tensor:
        with torch.no_grad():
            current_log_return = params.mu.new_zeros((num_samples,))
            peak_log_return = params.mu.new_zeros((num_samples,))
            current_drawdown_days = params.mu.new_zeros((num_samples,), dtype=torch.long)
            maximum_drawdown_days = params.mu.new_zeros((num_samples,), dtype=torch.long)
            var_i = params.var_0.repeat((num_samples,))
            for _ in range(num_days):
                x_i = torch.normal(params.mu, torch.sqrt(var_i))
                current_log_return += x_i
                peak_log_return = torch.maximum(peak_log_return, current_log_return)
                current_drawdown_days = torch.where(current_log_return < peak_log_return, current_drawdown_days + 1, torch.zeros_like(current_drawdown_days))
                maximum_drawdown_days = torch.maximum(maximum_drawdown_days, current_drawdown_days)
                squared_epsilon_i = (x_i - params.mu).square()
                var_i = params.omega + params.alpha * squared_epsilon_i + params.beta * var_i
            return maximum_drawdown_days


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
    data = torch.from_numpy(series_of_log_daily_return)
    fit_model(model, data, lr=0.5, num_iterations=200)
    model.print_params()

    # Compute covariance matrix of parameters
    theta_hat = torch.stack([
        model.unconstrained_mu.detach(),
        model.unconstrained_var_0.detach(),
        model.unconstrained_alpha.detach(),
        model.unconstrained_beta.detach(),
    ]).requires_grad_(True)
    hessian = torch.autograd.functional.hessian(lambda theta: model._negative_log_likelihood(model._params(
        theta[0],
        theta[1],
        theta[2],
        theta[3],
    ), data), theta_hat)
    print(f"Eigenvalues of Hessian: {torch.linalg.eigvalsh(hessian).tolist()}")
    cov: torch.Tensor = torch.linalg.inv(hessian)

    # Compute 90% confidence interval of annual long-run volatility
    se_of_unconstrained_var_0 = torch.sqrt(cov[1, 1])

    lower_bound_of_unconstrained_var_0: torch.Tensor = model.unconstrained_var_0 - stats.norm.ppf(0.95) * se_of_unconstrained_var_0
    upper_bound_of_unconstrained_var_0: torch.Tensor = model.unconstrained_var_0 - stats.norm.ppf(0.05) * se_of_unconstrained_var_0
    print(f"90% confidence interval of annual long-run volatility: [{torch.sqrt(F.softplus(lower_bound_of_unconstrained_var_0)).item():.2f}, {torch.sqrt(F.softplus(upper_bound_of_unconstrained_var_0)).item():.2f}]")

    # Compute 80% confidence interval of mean of log annual return
    se_of_unconstrained_mu = torch.sqrt(cov[0, 0])

    lower_bound_of_unconstrained_mu: torch.Tensor = model.unconstrained_mu - stats.norm.ppf(0.90) * se_of_unconstrained_mu
    upper_bound_of_unconstrained_mu: torch.Tensor = model.unconstrained_mu - stats.norm.ppf(0.10) * se_of_unconstrained_mu
    print(f"80% confidence interval of mean of log annual return: [{lower_bound_of_unconstrained_mu.item():.2f}, {upper_bound_of_unconstrained_mu.item():.2f}]")

    # Compute 50% confidence interval of mean of log annual return
    sublower_bound_of_unconstrained_mu: torch.Tensor = model.unconstrained_mu - stats.norm.ppf(0.75) * se_of_unconstrained_mu
    subupper_bound_of_unconstrained_mu: torch.Tensor = model.unconstrained_mu - stats.norm.ppf(0.25) * se_of_unconstrained_mu
    print(f"50% confidence interval of mean of log annual return: [{sublower_bound_of_unconstrained_mu.item():.2f}, {subupper_bound_of_unconstrained_mu.item():.2f}]")

    # Compute near-worst-case annual return
    samples_of_near_worst_case_log_annual_return_1: npt.NDArray[np.float64] = model._sample_log_annual_return(model._params(
        lower_bound_of_unconstrained_mu,
        upper_bound_of_unconstrained_var_0,
        model.unconstrained_alpha,
        model.unconstrained_beta,
    ), 10000).detach().numpy()
    samples_of_near_worst_case_log_annual_return_2: npt.NDArray[np.float64] = model._sample_log_annual_return(model._params(
        lower_bound_of_unconstrained_mu,
        lower_bound_of_unconstrained_var_0,
        model.unconstrained_alpha,
        model.unconstrained_beta,
    ), 10000).detach().numpy()

    def cdf_of_near_worst_case_annual_return(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cdf1: npt.NDArray[np.float64] = (samples_of_near_worst_case_log_annual_return_1[:, None] <= np.log(x)[None, :]).mean(axis=0)
        cdf2: npt.NDArray[np.float64] = (samples_of_near_worst_case_log_annual_return_2[:, None] <= np.log(x)[None, :]).mean(axis=0)
        return np.maximum(cdf1, cdf2)

    # Compute subnear-worst-case annual return
    samples_of_subnear_worst_case_log_annual_return_1: npt.NDArray[np.float64] = model._sample_log_annual_return(model._params(
        sublower_bound_of_unconstrained_mu,
        upper_bound_of_unconstrained_var_0,
        model.unconstrained_alpha,
        model.unconstrained_beta,
    ), 10000).detach().numpy()
    samples_of_subnear_worst_case_log_annual_return_2: npt.NDArray[np.float64] = model._sample_log_annual_return(model._params(
        sublower_bound_of_unconstrained_mu,
        lower_bound_of_unconstrained_var_0,
        model.unconstrained_alpha,
        model.unconstrained_beta,
    ), 10000).detach().numpy()

    def cdf_of_subnear_worst_case_annual_return(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cdf1: npt.NDArray[np.float64] = (samples_of_subnear_worst_case_log_annual_return_1[:, None] <= np.log(x)[None, :]).mean(axis=0)
        cdf2: npt.NDArray[np.float64] = (samples_of_subnear_worst_case_log_annual_return_2[:, None] <= np.log(x)[None, :]).mean(axis=0)
        return np.maximum(cdf1, cdf2)

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

    # Compute near-worst-case MDD days in 10 years
    samples_of_near_worst_case_mdd_days_in_10_years: npt.NDArray[np.long] = model._sample_mdd_days(model._params(
        lower_bound_of_unconstrained_mu,
        upper_bound_of_unconstrained_var_0,
        model.unconstrained_alpha,
        model.unconstrained_beta,
    ), 10000, 252*10).detach().numpy()

    def cdf_of_near_worst_case_mdd_days_in_10_years(x: npt.NDArray[np.long]) -> npt.NDArray[np.float64]:
        cdf: npt.NDArray[np.float64] = (samples_of_near_worst_case_mdd_days_in_10_years[:, None] <= x[None, :]).mean(axis=0, dtype=np.float64)
        return cdf

    # Plot near-worst-case MDD days in 10 years
    plt.figure()
    x = np.arange(0, 252*10+1, dtype=np.long)
    y = cdf_of_near_worst_case_mdd_days_in_10_years(x)
    plt.plot(x, y, label="Near-worst-case")
    plt.xlabel("MDD days in 10 years")
    plt.xlim(0, 252*10)
    plt.ylabel("Cumulative probability")
    plt.ylim(0, 1)
    plt.legend()
    if args.path_to_plot is not None:
        plt.savefig(args.path_to_plot.replace(".png", "_mddd.png"))
    else:
        plt.show()


if __name__ == "__main__":
    main()
