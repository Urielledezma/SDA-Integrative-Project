"""Fit the shape of the arrival curve to the Kaggle flight-prices data.

    python calibrate.py --csv path/to/itineraries.csv [--route LAX-JFK]

The brief asks for the arrival curve to be re-estimated from a real market with a fixed
capacity and a deadline, using the Expedia flight-prices data as a stand-in, and says to
work with a sample of one or a few routes because the file is large.  This reads it in
chunks, keeps one route, and writes the fitted parameters to results/calibration.json
together with everything needed to judge whether the fit means anything.

The curve fitted is the one the simulator uses,

    lambda(u) = A * exp(B * u),      u = (T - days_to_departure) / T

which is linear in its parameters once logged, so it is fitted by least squares on
log(sales per day) against u.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from dtp import config  # noqa: E402

RESULTS = HERE / "results"
WANTED = ["flightDate", "searchDate", "startingAirport", "destinationAirport",
          "seatsRemaining", "totalFare"]


def load_route(csv_path, route, chunk_rows=500_000, max_chunks=None):
    """Read the file in chunks and keep the rows for one route."""
    origin, destination = route.split("-")
    kept, scanned = [], 0
    reader = pd.read_csv(csv_path, usecols=WANTED, chunksize=chunk_rows)
    for i, chunk in enumerate(reader):
        scanned += len(chunk)
        match = chunk[(chunk.startingAirport == origin)
                      & (chunk.destinationAirport == destination)]
        if len(match):
            kept.append(match)
        if max_chunks is not None and i + 1 >= max_chunks:
            break
    if not kept:
        raise SystemExit(f"no rows found for {route}; check the airport codes")
    frame = pd.concat(kept, ignore_index=True)
    print(f"scanned {scanned:,} rows, kept {len(frame):,} for {route}")
    return frame


def daily_sales(frame):
    """Turn the panel into tickets sold per day to departure, for each departure.

    Seats remaining is observed once per search date per departure, so the sales on a day
    are the fall in seats remaining between that day and the previous one.  Only falls
    count, since a rise means inventory was added back or a different fare class was
    being quoted, and neither is a sale.
    """
    frame = frame.copy()
    frame["flightDate"] = pd.to_datetime(frame.flightDate)
    frame["searchDate"] = pd.to_datetime(frame.searchDate)
    frame["days_to_departure"] = (frame.flightDate - frame.searchDate).dt.days
    frame = frame[(frame.days_to_departure >= 0)
                  & (frame.days_to_departure <= config.SELLING_DAYS)]

    panel = (frame.groupby(["flightDate", "days_to_departure"])
             .agg(seats=("seatsRemaining", "min"), fare=("totalFare", "mean"))
             .reset_index()
             .sort_values(["flightDate", "days_to_departure"], ascending=[True, False]))
    panel["sold"] = -panel.groupby("flightDate").seats.diff()
    panel = panel[panel.sold.notna() & (panel.sold > 0)]
    return panel


def fit(panel):
    """Least squares of log(sales) on the fraction of the window elapsed."""
    days = config.SELLING_DAYS
    rate = (panel.groupby("days_to_departure")
            .agg(sold=("sold", "mean"), observations=("sold", "size"))
            .reset_index())
    rate["elapsed"] = (days - rate.days_to_departure) / days
    usable = rate[rate.sold > 0]
    if len(usable) < 5:
        raise SystemExit("fewer than five days with positive sales; the route is too thin")
    slope, intercept = np.polyfit(usable.elapsed, np.log(usable.sold), 1)
    predicted = np.exp(intercept + slope * usable.elapsed)
    residual = np.log(usable.sold) - np.log(predicted)
    r_squared = 1.0 - residual.var() / np.log(usable.sold).var()
    return float(np.exp(intercept)), float(slope), float(r_squared), rate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="path to itineraries.csv")
    ap.add_argument("--route", default="LAX-JFK", help="origin-destination, e.g. LAX-JFK")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="stop after this many chunks, for a quick look")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    frame = load_route(args.csv, args.route, max_chunks=args.max_chunks)

    seats = frame.seatsRemaining
    print(f"\nseats remaining: min {seats.min()}, median {seats.median()}, max {seats.max()}")
    if seats.max() <= 10:
        print("  NOTE: this column is capped in the source data, so it reports seats left")
        print("  at the quoted fare rather than on the aircraft. The shape of the decline is")
        print("  still informative about when demand arrives, while its level is not a count")
        print("  of tickets, and the report says so rather than treating A as a real rate.")

    panel = daily_sales(frame)
    print(f"departures {panel.flightDate.nunique()}, "
          f"day observations with a fall in seats {len(panel):,}")

    a, b, r_squared, rate = fit(panel)
    scale = config.CAPACITY / max(1.0, float(rate.sold.sum()))
    print(f"\nfitted lambda(u) = A exp(B u)")
    print(f"  B (shape)   {b:.4f}      how much the rate rises across the window")
    print(f"  A (level)   {a:.4f}      before rescaling")
    print(f"  R squared   {r_squared:.3f}  of log sales on elapsed fraction")
    print(f"\nthe level is rescaled so the window's expected arrivals match the venue, "
          f"because the data's level is a fare-class count and not a ticket count")
    elapsed = np.arange(config.SELLING_DAYS) / (config.SELLING_DAYS - 1)
    raw_total = float(np.sum(a * np.exp(b * elapsed)))
    a_scaled = a * (config.CAPACITY * 1.4) / raw_total
    print(f"  A (rescaled) {a_scaled:.4f}, giving {config.CAPACITY * 1.4:,.0f} expected "
          f"arrivals against {config.CAPACITY:,} seats")

    rate.to_csv(RESULTS / "calibration_rate.csv", index=False)
    out = {"route": args.route, "rows": int(len(frame)),
           "departures": int(panel.flightDate.nunique()),
           "arrival_a_raw": a, "arrival_a": a_scaled, "arrival_b": b,
           "r_squared": r_squared,
           "seats_column_capped": bool(seats.max() <= 10),
           "expected_arrivals": config.CAPACITY * 1.4}
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'calibration.json'}")
    print("put ARRIVAL_A and ARRIVAL_B from that file into src/dtp/config.py")


if __name__ == "__main__":
    main()
