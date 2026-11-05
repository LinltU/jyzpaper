"""Command-line entry point for the archived local training/prediction code."""
import argparse
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True, help="Private INI containing QUERY.devices")
    cli.add_argument("--data-folder", type=Path, default=Path("output"))
    cli.add_argument("--device-index", type=int, required=True, help="Zero-based index in QUERY.devices")
    actions = cli.add_subparsers(dest="action", required=True)
    train = actions.add_parser("train")
    train.add_argument("--save", type=Path, required=True, help="Private output model .pkl")
    predict = actions.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--temp", type=float, required=True)
    predict.add_argument("--hum", type=float, required=True)
    predict.add_argument("--history", type=float, nargs="+", required=True,
                         help="Past Measure values, oldest first")
    args = cli.parse_args()
    # Defer imports so --help works without the optional ML dependencies.
    from train import DeviceModelTrainer
    trainer = DeviceModelTrainer(str(args.config), data_folder=str(args.data_folder))
    if not 0 <= args.device_index < len(trainer.devices):
        cli.error("--device-index is outside QUERY.devices")
    if args.action == "train":
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.train_model(args.device_index, model_type="dlinear")
        trainer.save_model(str(args.save))
    else:
        if not args.model.is_file():
            cli.error("Model file does not exist")
        trainer.load_model(str(args.model))
        if trainer.model_type != "dlinear":
            cli.error("This entry point requires a dlinear model")
        print(trainer.predict(args.temp, args.hum, args.history))


if __name__ == "__main__":
    main()
