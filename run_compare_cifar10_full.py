import copy
import datetime
import os
import sys

from config import CONFIG
from csahfl_server import CSAHFLServer
from eafl_server import EAFLServer
from experiment_environment import create_environment
from run_compare_baselines import Logger, plot_results, run_experiment
from server import Server


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "baselines")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"compare_baselines_cifar10_full_{timestamp}.log")
    logger = Logger(log_filename)
    sys.stdout = logger

    original_config = copy.deepcopy(CONFIG)
    try:
        base_config = {
            "global_rounds": CONFIG.get("global_rounds", 500),
            "dataset": "cifar10",
            "input_channels": 3,
            "num_classes": 10,
            "model_name": CONFIG.get("model_name", "resnet18"),
            "device": CONFIG.get("device", "cpu"),
            "eafl_clustering_max_batches": CONFIG.get("eafl_clustering_max_batches", 2),
        }

        print(">>> Running full baseline comparison on CIFAR-10 only")
        print(f"Logging to {log_filename}")
        print(f"Using device: {base_config['device']}")

        env_config = copy.deepcopy(CONFIG)
        env_config.update(base_config)
        environment = create_environment(env_config, seed=env_config.get("seed", 42), device=base_config["device"])

        results = {}

        print("\n--- Running EAFL on CIFAR-10 ---")
        results["EAFL"] = run_experiment(
            f"EAFL_cifar10_full_{timestamp}",
            EAFLServer,
            copy.deepcopy(base_config),
            environment=environment,
        )

        print("\n--- Running CSAHFL on CIFAR-10 ---")
        results["CSAHFL"] = run_experiment(
            f"CSAHFL_cifar10_full_{timestamp}",
            CSAHFLServer,
            copy.deepcopy(base_config),
            environment=environment,
        )

        print("\n--- Running Ours on CIFAR-10 ---")
        ours_config = copy.deepcopy(base_config)
        ours_config["clustering_mode"] = "soft"
        ours_config["aggregation_strategy"] = "hide_adf"
        results["Ours"] = run_experiment(
            f"Ours_cifar10_full_{timestamp}",
            Server,
            ours_config,
            environment=environment,
        )

        plot_results(
            results,
            "Ours vs EAFL vs CSAHFL - CIFAR-10 Full",
            f"ours_vs_baselines_cifar10_full_{timestamp}.png",
        )

        print("\n>>> CIFAR-10 full comparison finished")
        for name, accuracies in results.items():
            if accuracies:
                print(f"{name}: rounds={len(accuracies)}, final={accuracies[-1]:.4f}, best={max(accuracies):.4f}")
            else:
                print(f"{name}: no results")
    finally:
        CONFIG.clear()
        CONFIG.update(original_config)
        sys.stdout = logger.terminal
        logger.log.close()
        print(f"Log saved to {log_filename}")


if __name__ == "__main__":
    main()
