import argparse
import datetime as dt
import json
import math

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


log_doubled_pi = math.log(2 * math.pi)


def fit_model(model: nn.Module, data: torch.Tensor, lr: float, num_iterations: int) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        loss: torch.Tensor = model(data)
        loss.backward()
        optimizer.step()
        print(f"Current loss: {loss.item():.6f}")


sqrt_252: np.float64 = np.sqrt(252)


def common() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.datetime64], argparse.Namespace]:
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
        "--supplementary-leverage",
        metavar="SL",
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
    parser.add_argument(
        "--path-to-plot",
        metavar="path/to/plot",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Read JSON
    with open(args.path_to_json, "r") as fp:
        table: list[list[str]] = json.load(fp)
    series_of_price: npt.NDArray[np.float64] = np.array([row[5].replace(",", "") for row in table], dtype="float64")
    series_of_date: npt.NDArray[np.datetime64] = np.array([dt.datetime.strptime(row[0], "%b %d, %Y") for row in table], dtype="datetime64[D]")

    # Get leverage
    leverage = np.float64(args.leverage)
    print(f"Leverage: {leverage}")

    # Compute series of daily return
    series_of_daily_return = (series_of_price[:-1] / series_of_price[1:] - 1) * leverage + 1
    series_of_date = series_of_date[:-1]

    if args.path_to_supplementary_json is not None:
        # Read supplementary JSON
        with open(args.path_to_supplementary_json, "r") as supplementary_fp:
            supplementary_table: list[list[str]] = json.load(supplementary_fp)
        supplementary_series_of_price: npt.NDArray[np.float64] = np.array([supplementary_row[5].replace(",", "") for supplementary_row in supplementary_table], dtype="float64")
        supplementary_series_of_date: npt.NDArray[np.datetime64] = np.array([dt.datetime.strptime(supplementary_row[0], "%b %d, %Y") for supplementary_row in supplementary_table], dtype="datetime64[D]")

        # Filter supplementary series of price and date
        supplementary_series_of_price: npt.NDArray[np.float64] = supplementary_series_of_price[supplementary_series_of_date < series_of_date[-1]]
        supplementary_series_of_date: npt.NDArray[np.datetime64] = supplementary_series_of_date[supplementary_series_of_date < series_of_date[-1]]

        # Get supplementary leverage
        supplementary_leverage = np.float64(args.supplementary_leverage)
        print(f"Supplementary leverage: {supplementary_leverage}")

        # Compute supplementary series of daily return
        supplementary_series_of_daily_return = (supplementary_series_of_price[:-1] / supplementary_series_of_price[1:] - 1) * supplementary_leverage + 1
        supplementary_series_of_date = supplementary_series_of_date[:-1]

        # Merge series of daily return and supplementary series of daily return
        series_of_daily_return = np.concatenate((series_of_daily_return, supplementary_series_of_daily_return))
        series_of_date = np.concatenate((series_of_date, supplementary_series_of_date))

    # Check quality of series of daily return
    series_of_abnormal_daily_return = series_of_daily_return[(series_of_daily_return < 0.6) | (series_of_daily_return > 1.8)]
    series_of_abnormal_date = series_of_date[(series_of_daily_return < 0.6) | (series_of_daily_return > 1.8)]
    whitelist_of_abnormal_date = np.array([
        np.datetime64("2025-04-09"),
        np.datetime64("2008-10-28"),
        np.datetime64("2001-01-03"),
        np.datetime64("2000-10-19"),
        np.datetime64("1987-10-19"),
        np.datetime64("1933-03-15"),
    ])
    series_of_abnormal_daily_return = series_of_abnormal_daily_return[~np.isin(series_of_abnormal_date, whitelist_of_abnormal_date)]
    series_of_abnormal_date = series_of_abnormal_date[~np.isin(series_of_abnormal_date, whitelist_of_abnormal_date)]
    for abnormal_daily_return, abnormal_date in zip(series_of_abnormal_daily_return, series_of_abnormal_date):
        print(f"Abnormal daily return: {abnormal_daily_return:.2f} on {np.datetime_as_string(abnormal_date, unit='D')}")

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

    return series_of_daily_return, series_of_date, args
