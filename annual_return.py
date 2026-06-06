#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from common import common, sqrt_252


def main() -> None:
    series_of_daily_return, _, args = common()

    # Compute series of log daily return
    series_of_log_daily_return: npt.NDArray[np.float64] = np.log(series_of_daily_return)

    # Compute 90% confidence interval of annual volatility
    daily_volatility: np.float64 = series_of_log_daily_return.std(ddof=1)

    lower_bound_of_daily_volatility: np.float64 = np.sqrt((series_of_log_daily_return.shape[0]-1) / stats.chi2.ppf(0.95, df=series_of_log_daily_return.shape[0]-1)) * daily_volatility
    upper_bound_of_daily_volatility: np.float64 = np.sqrt((series_of_log_daily_return.shape[0]-1) / stats.chi2.ppf(0.05, df=series_of_log_daily_return.shape[0]-1)) * daily_volatility
    print(f"90% confidence interval of annual volatility: [{lower_bound_of_daily_volatility * sqrt_252:.2f}, {upper_bound_of_daily_volatility * sqrt_252:.2f}]")

    # Compute 80% confidence interval of annual drift minus halved squared annual volatility
    daily_drift_minus_halved_squared_daily_volatility: np.float64 = series_of_log_daily_return.mean()

    lower_bound_of_daily_drift_minus_halved_squared_daily_volatility: np.float64 = daily_drift_minus_halved_squared_daily_volatility - stats.t.ppf(0.90, df=series_of_log_daily_return.shape[0]-1) * daily_volatility / np.sqrt(series_of_log_daily_return.shape[0])
    upper_bound_of_daily_drift_minus_halved_squared_daily_volatility: np.float64 = daily_drift_minus_halved_squared_daily_volatility - stats.t.ppf(0.10, df=series_of_log_daily_return.shape[0]-1) * daily_volatility / np.sqrt(series_of_log_daily_return.shape[0])
    print(f"80% confidence interval of annual drift minus halved squared annual volatility: [{lower_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252:.2f}, {upper_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252:.2f}]")

    # Compute 50% confidence interval of annual drift minus halved squared annual volatility
    sublower_bound_of_daily_drift_minus_halved_squared_daily_volatility: np.float64 = daily_drift_minus_halved_squared_daily_volatility - stats.t.ppf(0.75, df=series_of_log_daily_return.shape[0]-1) * daily_volatility / np.sqrt(series_of_log_daily_return.shape[0])
    subupper_bound_of_daily_drift_minus_halved_squared_daily_volatility: np.float64 = daily_drift_minus_halved_squared_daily_volatility - stats.t.ppf(0.25, df=series_of_log_daily_return.shape[0]-1) * daily_volatility / np.sqrt(series_of_log_daily_return.shape[0])
    print(f"50% confidence interval of annual drift minus halved squared annual volatility: [{sublower_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252:.2f}, {subupper_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252:.2f}]")

    # Compute 90% Monte Carlo interval of MDD in 3 years
    def sample_negative_log_one_minus_mdd(num_samples: int, num_days: int) -> npt.NDArray[np.float64]:
        current_log_return: npt.NDArray[np.float64] = np.zeros((num_samples,))
        peak_log_return: npt.NDArray[np.float64] = np.zeros((num_samples,))
        negative_log_one_minus_mdd: npt.NDArray[np.float64] = np.zeros((num_samples,))

        for _ in range(num_days):
            x_i = np.random.normal(lower_bound_of_daily_drift_minus_halved_squared_daily_volatility, upper_bound_of_daily_volatility, size=(num_samples,))
            current_log_return += x_i
            peak_log_return: npt.NDArray[np.float64] = np.maximum(peak_log_return, current_log_return)
            negative_log_one_minus_mdd: npt.NDArray[np.float64] = np.maximum(negative_log_one_minus_mdd, peak_log_return - current_log_return)
        return negative_log_one_minus_mdd
    samples_of_negative_log_one_minus_mdd = sample_negative_log_one_minus_mdd(100000, 252*3)

    lower_bound_of_negative_log_one_minus_mdd = np.percentile(samples_of_negative_log_one_minus_mdd, 5)
    upper_bound_of_negative_log_one_minus_mdd = np.percentile(samples_of_negative_log_one_minus_mdd, 95)
    print(f"90% Monte Carlo interval of MDD in 3 years: [{1 - np.exp(-lower_bound_of_negative_log_one_minus_mdd):.2f}, {1 - np.exp(-upper_bound_of_negative_log_one_minus_mdd):.2f}]")

    # Compute 50% Monte Carlo interval of MDD in 3 years
    sublower_bound_of_negative_log_one_minus_mdd = np.percentile(samples_of_negative_log_one_minus_mdd, 25)
    subupper_bound_of_negative_log_one_minus_mdd = np.percentile(samples_of_negative_log_one_minus_mdd, 75)
    print(f"50% Monte Carlo interval of MDD in 3 years: [{1 - np.exp(-sublower_bound_of_negative_log_one_minus_mdd):.2f}, {1 - np.exp(-subupper_bound_of_negative_log_one_minus_mdd):.2f}]")

    # Compute near-worst-case annual return
    near_worst_case_annual_return_1 = stats.lognorm(s=upper_bound_of_daily_volatility*sqrt_252, scale=np.exp(lower_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252))
    near_worst_case_annual_return_2 = stats.lognorm(s=lower_bound_of_daily_volatility*sqrt_252, scale=np.exp(lower_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252))

    def cdf_of_near_worst_case_annual_return(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cdf1: npt.NDArray[np.float64] = near_worst_case_annual_return_1.cdf(x)
        cdf2: npt.NDArray[np.float64] = near_worst_case_annual_return_2.cdf(x)
        return np.maximum(cdf1, cdf2)

    # Compute subnear-worst-case annual return
    subnear_worst_case_annual_return_1 = stats.lognorm(s=upper_bound_of_daily_volatility*sqrt_252, scale=np.exp(sublower_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252))
    subnear_worst_case_annual_return_2 = stats.lognorm(s=lower_bound_of_daily_volatility*sqrt_252, scale=np.exp(sublower_bound_of_daily_drift_minus_halved_squared_daily_volatility * 252))

    def cdf_of_subnear_worst_case_annual_return(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cdf1: npt.NDArray[np.float64] = subnear_worst_case_annual_return_1.cdf(x)
        cdf2: npt.NDArray[np.float64] = subnear_worst_case_annual_return_2.cdf(x)
        return np.maximum(cdf1, cdf2)

    # Compute series of annual return
    series_of_annual_return: npt.NDArray[np.float64] = series_of_daily_return[series_of_daily_return.shape[0] % 252:].reshape((-1, 252)).prod(axis=1)

    # Plot annual return
    x = np.linspace(0, 3, 1000)
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
