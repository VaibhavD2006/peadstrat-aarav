import pytest
import mlflow
from aria.research.tracker import ExperimentTracker

def test_tracker_logs_metrics(tmp_path):
    db_path = str(tmp_path / "mlruns.db").replace("\\", "/")
    uri = f"sqlite:///{db_path}"
    tracker = ExperimentTracker(experiment_name="ARIA_test", tracking_uri=uri)
    with tracker.start_run("E01_test"):
        tracker.log_params({"signals": "ESQS", "hold_days": 10})
        tracker.log_metrics({"sharpe": 0.85, "max_drawdown": -0.07})
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("ARIA_test")
    assert experiment is not None
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) == 1
    assert runs[0].data.metrics["sharpe"] == pytest.approx(0.85)

def test_tracker_creates_experiment(tmp_path):
    db_path = str(tmp_path / "mlruns2.db").replace("\\", "/")
    uri = f"sqlite:///{db_path}"
    tracker = ExperimentTracker(experiment_name="NEW_EXP", tracking_uri=uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("NEW_EXP")
    assert exp is not None
