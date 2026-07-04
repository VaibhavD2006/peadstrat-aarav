"""ARIA Research CLI."""
import argparse
import sys

def run_backtest(args: argparse.Namespace) -> None:
    signals = [s.strip() for s in args.signals.split(",")]
    print(f"[ARIA] run-backtest")
    print(f"  signals   : {signals}")
    print(f"  hold_days : {args.hold_days}")
    print(f"  start     : {args.start}")
    print(f"  end       : {args.end}")
    print("TODO: wire up full pipeline: data ingestion -> signals -> backtest -> performance")

def event_study(args: argparse.Namespace) -> None:
    print(f"[ARIA] event-study")
    print(f"  signal : {args.signal}")
    print(f"  start  : {args.start}")
    print(f"  end    : {args.end}")
    print("TODO: wire up event study pipeline")

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aria",
        description="ARIA Strategy Research CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("run-backtest", help="Run strategy backtest")
    bt.add_argument("--signals",   default="ESQS,RMV",  help="Comma-separated signal names")
    bt.add_argument("--hold-days", type=int, default=10, help="Holding period in business days")
    bt.add_argument("--start",     default="2014-01-01", help="Backtest start date YYYY-MM-DD")
    bt.add_argument("--end",       default="2024-12-31", help="Backtest end date YYYY-MM-DD")

    es = sub.add_parser("event-study", help="Run single-signal event study")
    es.add_argument("--signal", default="ESQS",       help="Signal name")
    es.add_argument("--start",  default="2018-01-01", help="Start date YYYY-MM-DD")
    es.add_argument("--end",    default="2023-12-31", help="End date YYYY-MM-DD")

    args = parser.parse_args()
    if args.command == "run-backtest":
        run_backtest(args)
    elif args.command == "event-study":
        event_study(args)

if __name__ == "__main__":
    main()
