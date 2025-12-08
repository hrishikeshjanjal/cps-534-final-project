from src.config_loader import load_config
from src.data.activity_parser import extract_activity_from_ann_features


def main():
    cfg = load_config("configs/config.yaml")
    ann_path = cfg["activity"]["ann_features_path"]
    out_path = cfg["activity"]["processed_activity_csv"]
    sample_period = int(cfg.get("sampling", {}).get("period_seconds", 10))

    extract_activity_from_ann_features(
        ann_features_path=ann_path,
        output_csv_path=out_path,
        sample_period_seconds=sample_period,
    )


if __name__ == "__main__":
    main()
