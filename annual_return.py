#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from common import common

sqrt_252: np.float64 = np.sqrt(252)


def main() -> None:
    series_of_daily_return, _, args = common()

    # Compute series of logarithm of daily return
    series_of_logarithm_of_daily_return: npt.NDArray[np.float64] = np.log(series_of_daily_return)

    # Compute 90% confidence interval of annual volatility
    daily_volatility: np.float64 = series_of_logarithm_of_daily_return.std(ddof=1)

    lower_bound_of_daily_volatility: np.float64 = np.sqrt((len(series_of_logarithm_of_daily_return) - 1) / stats.chi2.ppf(0.95, df=len(series_of_logarithm_of_daily_return)-1)) * daily_volatility  # type: ignore
    upper_bound_of_daily_volatility: np.float64 = np.sqrt((len(series_of_logarithm_of_daily_return) - 1) / stats.chi2.ppf(0.05, df=len(series_of_logarithm_of_daily_return)-1)) * daily_volatility  # type: ignore
    print(f"90% confidence interval of annual volatility: [{lower_bound_of_daily_volatility * sqrt_252:.2f}, {upper_bound_of_daily_volatility * sqrt_252:.2f}]")

    # Compute 80% confidence interval of annual drift minus annual volatility squared halved
    daily_drift_minus_daily_volatility_squared_halved: np.float64 = series_of_logarithm_of_daily_return.mean()

    lower_bound_of_daily_drift_minus_daily_volatility_squared_halved: np.float64 = daily_drift_minus_daily_volatility_squared_halved - stats.t.ppf(0.90, df=len(series_of_logarithm_of_daily_return)-1) * daily_volatility / np.sqrt(len(series_of_logarithm_of_daily_return))
    upper_bound_of_daily_drift_minus_daily_volatility_squared_halved: np.float64 = daily_drift_minus_daily_volatility_squared_halved - stats.t.ppf(0.10, df=len(series_of_logarithm_of_daily_return)-1) * daily_volatility / np.sqrt(len(series_of_logarithm_of_daily_return))
    print(f"80% confidence interval of annual drift minus annual volatility squared halved: [{lower_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252:.2f}, {upper_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252:.2f}]")

    # Compute 50% confidence interval of annual drift minus annual volatility squared halved
    sublower_bound_of_daily_drift_minus_daily_volatility_squared_halved: np.float64 = daily_drift_minus_daily_volatility_squared_halved - stats.t.ppf(0.75, df=len(series_of_logarithm_of_daily_return)-1) * daily_volatility / np.sqrt(len(series_of_logarithm_of_daily_return))
    subupper_bound_of_daily_drift_minus_daily_volatility_squared_halved: np.float64 = daily_drift_minus_daily_volatility_squared_halved - stats.t.ppf(0.25, df=len(series_of_logarithm_of_daily_return)-1) * daily_volatility / np.sqrt(len(series_of_logarithm_of_daily_return))
    print(f"50% confidence interval of annual drift minus annual volatility squared halved: [{sublower_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252:.2f}, {subupper_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252:.2f}]")

    # Compute 80% confidence interval of median of annual return
    lower_bound_of_median_of_annual_return: np.float64 = np.exp(lower_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252)
    upper_bound_of_median_of_annual_return: np.float64 = np.exp(upper_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252)
    print(f"80% confidence interval of median of annual return: [{lower_bound_of_median_of_annual_return:.2f}, {upper_bound_of_median_of_annual_return:.2f}]")

    # Compute 50% confidence interval of median of annual return
    sublower_bound_of_median_of_annual_return: np.float64 = np.exp(sublower_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252)
    subupper_bound_of_median_of_annual_return: np.float64 = np.exp(subupper_bound_of_daily_drift_minus_daily_volatility_squared_halved * 252)
    print(f"50% confidence interval of median of annual return: [{sublower_bound_of_median_of_annual_return:.2f}, {subupper_bound_of_median_of_annual_return:.2f}]")

    # Compute near-worst-case annual return
    near_worst_case_annual_return: stats.rv_continuous = stats.lognorm(s=daily_volatility*sqrt_252, scale=lower_bound_of_median_of_annual_return)  # type: ignore

    # Compute subnear-worst-case annual return
    subnear_worst_case_annual_return: stats.rv_continuous = stats.lognorm(s=daily_volatility*sqrt_252, scale=sublower_bound_of_median_of_annual_return)  # type: ignore

    # Compute series of annual return
    series_of_annual_return: npt.NDArray[np.float64] = series_of_daily_return[len(series_of_daily_return) % 252:].reshape((-1, 252)).prod(axis=1)

    # Plot worst-case annual return
    x = np.linspace(0, 3, 1000)
    y1 = near_worst_case_annual_return.cdf(x)
    plt.plot(x, y1, label="Near-worst-case Theoretical")
    y2 = subnear_worst_case_annual_return.cdf(x)
    plt.plot(x, y2, label="Subnear-worst-case Theoretical")
    plt.hist(series_of_annual_return, bins=len(series_of_annual_return), density=True, cumulative=True, alpha=0.5, label="Empirical")
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
