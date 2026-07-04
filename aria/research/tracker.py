import mlflow
from contextlib import contextmanager

class ExperimentTracker:
    def __init__(self, experiment_name: str = "ARIA",
                 tracking_uri: str = "sqlite:///mlruns.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str):
        with mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_params(self, params: dict) -> None:
        mlflow.log_params({str(k): str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
