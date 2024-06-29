#!/usr/bin/env python3

import argparse
import datetime as dt
import json

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-json",
        metavar="path/to/json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path-to-supplementary-json",
        metavar="path/to/supplementary/json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--leverage",
        metavar="L",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--last-date",
        metavar="YYYY-MM-DD",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--first-date",
        metavar="YYYY-MM-DD",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Read JSON
    with open(args.path_to_json, "r") as fp:
        table: list[list[str]] = json.load(fp)
    series_of_price: npt.NDArray[np.float64] = np.array([row[5].replace(",", "") for row in table], dtype="float64")
    series_of_date: npt.NDArray[np.datetime64] = np.array([dt.datetime.strptime(row[0], "%b %d, %Y") for row in table], dtype="datetime64[D]")

    # Read supplementary JSON
    if args.path_to_supplementary_json is None:
        supplementary_series_of_price: npt.NDArray[np.float64] = np.array([], dtype="float64")
        supplementary_series_of_date: npt.NDArray[np.datetime64] = np.array([], dtype="datetime64[D]")
    else:
        with open(args.path_to_supplementary_json, "r") as supplementary_fp:
            supplementary_table: list[list[str]] = json.load(supplementary_fp)
        supplementary_series_of_price: npt.NDArray[np.float64] = np.array([supplementary_row[5].replace(",", "") for supplementary_row in supplementary_table], dtype="float64")
        supplementary_series_of_date: npt.NDArray[np.datetime64] = np.array([dt.datetime.strptime(supplementary_row[0], "%b %d, %Y") for supplementary_row in supplementary_table], dtype="datetime64[D]")

    # Supplement series of price and date
    supplementary_series_of_price = supplementary_series_of_price[supplementary_series_of_date <= series_of_date[-1]]
    supplementary_series_of_date = supplementary_series_of_date[supplementary_series_of_date <= series_of_date[-1]]

    if len(supplementary_series_of_price) > 0:
        supplementary_series_of_price = supplementary_series_of_price[1:] * series_of_price[-1] / supplementary_series_of_price[0]
        supplementary_series_of_date = supplementary_series_of_date[1:]

    series_of_price = np.concatenate((series_of_price, supplementary_series_of_price))
    series_of_date = np.concatenate((series_of_date, supplementary_series_of_date))

    # Get leverage
    leverage = np.float64(args.leverage)
    print(f"Leverage: {leverage}")

    # Compute series of daily return
    series_of_daily_return = (series_of_price[:-1] / series_of_price[1:] - 1) * leverage + 1
    series_of_date = series_of_date[:-1]

    # Get last date
    if args.last_date is None:
        last_date = np.datetime64(series_of_date[0], "D")
    else:
        last_date = np.datetime64(args.last_date, "D")
    print(f"Last date: {np.datetime_as_string(last_date, unit='D')}")

    # Filter series of daily return by last date
    series_of_daily_return = series_of_daily_return[series_of_date <= last_date]
    series_of_date = series_of_date[series_of_date <= last_date]

    # Get first date
    if args.first_date is None:
        first_date = np.datetime64(series_of_date[-1], "D")
    else:
        first_date = np.datetime64(args.first_date, "D")
    print(f"First date: {np.datetime_as_string(first_date, unit='D')}")

    # Filter series of daily return by first date
    series_of_daily_return = series_of_daily_return[series_of_date >= first_date]
    series_of_date = series_of_date[series_of_date >= first_date]

    # Compute series of (logarithm of) annual return
    series_of_annual_return: npt.NDArray[np.float64] = series_of_daily_return[len(series_of_daily_return) % 252:].reshape((-1, 252)).prod(axis=1)
    series_of_logarithm_of_annual_return: npt.NDArray[np.float64] = np.log(series_of_annual_return)

    # Compute 90% confidence interval of annual volatility
    annual_volatility: np.float64 = series_of_logarithm_of_annual_return.std(ddof=1)

    lower_bound_of_annual_volatility: np.float64 = np.sqrt((len(series_of_logarithm_of_annual_return) - 1) / stats.chi2.ppf(0.95, df=len(series_of_logarithm_of_annual_return)-1)) * annual_volatility  # type: ignore
    upper_bound_of_annual_volatility: np.float64 = np.sqrt((len(series_of_logarithm_of_annual_return) - 1) / stats.chi2.ppf(0.05, df=len(series_of_logarithm_of_annual_return)-1)) * annual_volatility  # type: ignore
    print(f"90% confidence interval of annual volatility: [{lower_bound_of_annual_volatility}, {upper_bound_of_annual_volatility}]")

    # Compute 90% confidence interval of annual drift minus annual volatility squared halved
    annual_drift_minus_annual_volatility_squared_halved: np.float64 = series_of_logarithm_of_annual_return.mean()

    lower_bound_of_annual_drift_minus_annual_volatility_squared_halved: np.float64 = annual_drift_minus_annual_volatility_squared_halved - stats.t.ppf(0.95, df=len(series_of_logarithm_of_annual_return)-1) * annual_volatility / np.sqrt(len(series_of_logarithm_of_annual_return))
    upper_bound_of_annual_drift_minus_annual_volatility_squared_halved: np.float64 = annual_drift_minus_annual_volatility_squared_halved - stats.t.ppf(0.05, df=len(series_of_logarithm_of_annual_return)-1) * annual_volatility / np.sqrt(len(series_of_logarithm_of_annual_return))
    print(f"90% confidence interval of annual drift minus annual volatility squared halved: [{lower_bound_of_annual_drift_minus_annual_volatility_squared_halved}, {upper_bound_of_annual_drift_minus_annual_volatility_squared_halved}]")

    # Compute 90% confidence interval of median of annual return
    lower_bound_of_median_of_annual_return: np.float64 = np.exp(lower_bound_of_annual_drift_minus_annual_volatility_squared_halved)
    upper_bound_of_median_of_annual_return: np.float64 = np.exp(upper_bound_of_annual_drift_minus_annual_volatility_squared_halved)
    print(f"90% confidence interval of median of annual return: [{lower_bound_of_median_of_annual_return}, {upper_bound_of_median_of_annual_return}]")

    # Compute worst-case annual return
    worst_case_annual_return: stats.rv_continuous = stats.lognorm(s=annual_volatility, scale=lower_bound_of_median_of_annual_return)  # type: ignore

    # Plot worst-case annual return
    x = np.linspace(0, 3, 1000)
    y = worst_case_annual_return.cdf(x)
    plt.plot(x, y, label="Worst-case Theoretical")
    plt.hist(series_of_annual_return, bins=len(series_of_annual_return), density=True, cumulative=True, alpha=0.5, label="Empirical")
    plt.xlabel("Annual return")
    plt.xlim(0, 3)
    plt.ylabel("Cumulative probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
