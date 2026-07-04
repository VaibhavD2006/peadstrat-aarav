import pytest
from aria.research.ablation import AblationRunner, ExperimentSpec

def test_runner_stores_experiments():
    exps = [
        ExperimentSpec("E01_ESQS_only", ["ESQS_z"], {"ESQS_z": 1.0}),
        ExperimentSpec("E02_RMV_only",  ["RMV_z"],  {"RMV_z": 1.0}),
        ExperimentSpec("E03_combined",  ["ESQS_z", "RMV_z"], {"ESQS_z": 0.5, "RMV_z": 0.5}),
    ]
    runner = AblationRunner(experiments=exps)
    assert len(runner.experiments) == 3
    assert runner.experiments[0].name == "E01_ESQS_only"

def test_record_and_retrieve():
    runner = AblationRunner([ExperimentSpec("E01", ["ESQS_z"], {"ESQS_z": 1.0})])
    runner.record("E01", {"sharpe": 0.8, "max_drawdown": -0.07})
    assert runner.results["E01"]["sharpe"] == 0.8

def test_summary_table_includes_all_experiments():
    exps = [
        ExperimentSpec("E01", ["ESQS_z"], {"ESQS_z": 1.0}),
        ExperimentSpec("E02", ["RMV_z"],  {"RMV_z": 1.0}),
    ]
    runner = AblationRunner(exps)
    runner.record("E01", {"sharpe": 0.9})
    table = runner.summary_table()
    assert len(table) == 2
    e01_row = next(r for r in table if r["experiment"] == "E01")
    assert e01_row["sharpe"] == 0.9

def test_summary_table_no_results_shows_experiment_name():
    runner = AblationRunner([ExperimentSpec("EMPTY", ["ESQS_z"], {"ESQS_z": 1.0})])
    table = runner.summary_table()
    assert table[0]["experiment"] == "EMPTY"
