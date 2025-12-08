import argparse

from src.eval.metrics import load_log, compute_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Smart Desk simulation log")
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/simulation_log.csv",
        help="Path to simulation log CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_log(args.log_path)
    metrics = compute_metrics(df)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
