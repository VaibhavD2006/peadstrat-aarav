import pytest
import polars as pl
from aria.research.ablation import AblationRunner, ExperimentSpec, ABLATION_MATRIX


def test_default_matrix_has_expected_experiments():
    runner = AblationRunner()
    assert len(runner.experiments) == len(ABLATION_MATRIX)


def test_matrix_names_are_unique():
    names = [e.name for e in ABLATION_MATRIX]
    assert len(names) == len(set(names))


def test_matrix_covers_e01_through_e15():
    names = {e.name for e in ABLATION_MATRIX}
    for i in range(1, 10):
        prefix = f"E0{i}_"
        assert any(n.startswith(prefix) for n in names), f"Missing experiment starting with {prefix}"
    for i in range(10, 16):
        prefix = f"E{i}_"
        assert any(n.startswith(prefix) for n in names), f"Missing experiment starting with {prefix}"


def test_regime_filter_experiments_flagged():
    regime_exps = [e for e in ABLATION_MATRIX if e.regime_filter]
    assert len(regime_exps) >= 2  # E08 and E09


def test_ivrs_multiplier_experiments_flagged():
    ivrs_exps = [e for e in ABLATION_MATRIX if e.ivrs_multiplier]
    assert len(ivrs_exps) >= 2  # E04, E07, E08


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


def test_summary_df_returns_polars_dataframe():
    runner = AblationRunner()
    runner.record("E01_ESQS_only", {"sharpe": 1.2, "max_drawdown": -0.05})
    df = runner.summary_df()
    assert isinstance(df, pl.DataFrame)
    assert "experiment" in df.columns
    assert df.shape[0] == len(ABLATION_MATRIX)


def test_best_by_returns_highest_sharpe():
    exps = [
        ExperimentSpec("E01", ["ESQS_z"], {"ESQS_z": 1.0}),
        ExperimentSpec("E02", ["RMV_z"],  {"RMV_z": 1.0}),
        ExperimentSpec("E03", ["ESQS_z", "RMV_z"], {"ESQS_z": 0.5, "RMV_z": 0.5}),
    ]
    runner = AblationRunner(exps)
    runner.record("E01", {"sharpe": 0.8})
    runner.record("E02", {"sharpe": 1.5})
    runner.record("E03", {"sharpe": 1.2})
    best = runner.best_by("sharpe")
    assert best is not None
    assert best.name == "E02"


def test_best_by_returns_none_when_no_results():
    runner = AblationRunner()
    assert runner.best_by("sharpe") is None
