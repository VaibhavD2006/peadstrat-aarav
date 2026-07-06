from dataclasses import dataclass
from typing import Optional
import polars as pl


@dataclass
class ExperimentSpec:
    name: str
    signals: list[str]
    weights: dict[str, float]
    hold_days: int = 10
    regime_filter: bool = False
    ivrs_multiplier: bool = False
    bsq_filter: bool = False
    beta_neutral: bool = False
    vol_target: float = 0.0   # 0 = disabled; >0 = target annualised portfolio vol
    stop_loss_pct: float = 0.0   # 0 = disabled; e.g. 0.10 = 10% per-position stop
    use_revision_weight: bool = False  # True → apply revision_dir as weight multiplier
    concurrent_vol_adjust: bool = False  # True → divide vol target by sqrt(n concurrent cohorts)
    rho_cross_cohort: float = 0.0  # Cross-cohort correlation; 0 = independent (E26 formula), -0.05 = empirical L/S
    min_sue_z: float = 0.0  # Drop trades where |SUE_z| < this threshold (0 = disabled)
    notes: str = ""


# E01-E10 canonical ablation matrix
# Note: RMV_z is replaced at runtime by FTS_z (SEC EDGAR filing timeliness signal)
# since analyst revision data is not available in the free SimFin tier.
ABLATION_MATRIX: list[ExperimentSpec] = [
    ExperimentSpec(
        "E01_ESQS_only",
        signals=["ESQS_z"],
        weights={"ESQS_z": 1.0},
        notes="Baseline: earnings quality only (YoY consensus)",
    ),
    ExperimentSpec(
        "E02_FTS_only",
        signals=["FTS_z"],
        weights={"FTS_z": 1.0},
        notes="Baseline: EDGAR filing timeliness signal only",
    ),
    ExperimentSpec(
        "E03_ESQS_FTS",
        signals=["ESQS_z", "FTS_z"],
        weights={"ESQS_z": 0.5, "FTS_z": 0.5},
        notes="Core composite: earnings quality + filing timeliness",
    ),
    ExperimentSpec(
        "E04_IVRS_mult",
        signals=["ESQS_z", "FTS_z"],
        weights={"ESQS_z": 0.5, "FTS_z": 0.5},
        ivrs_multiplier=True,
        notes="E03 + IVRS position-size multiplier",
    ),
    ExperimentSpec(
        "E05_IFR_only",
        signals=["IFR_z"],
        weights={"IFR_z": 1.0},
        notes="Institutional flow residual standalone",
    ),
    ExperimentSpec(
        "E06_ESQS_IFR",
        signals=["ESQS_z", "IFR_z"],
        weights={"ESQS_z": 0.6, "IFR_z": 0.4},
        notes="Earnings quality + institutional flow",
    ),
    ExperimentSpec(
        "E07_full_no_regime",
        signals=["ESQS_z", "FTS_z", "IFR_z"],
        weights={"ESQS_z": 0.4, "FTS_z": 0.3, "IFR_z": 0.3},
        ivrs_multiplier=True,
        notes="All signals + IVRS, no regime filter",
    ),
    ExperimentSpec(
        "E08_full_regime",
        signals=["ESQS_z", "FTS_z", "IFR_z"],
        weights={"ESQS_z": 0.4, "FTS_z": 0.3, "IFR_z": 0.3},
        ivrs_multiplier=True,
        regime_filter=True,
        notes="Full stack with rolling HMM regime gate",
    ),
    ExperimentSpec(
        "E09_equal_weight",
        signals=["ESQS_z", "FTS_z", "IFR_z"],
        weights={"ESQS_z": 1/3, "FTS_z": 1/3, "IFR_z": 1/3},
        regime_filter=True,
        notes="Equal-weight all signals + regime filter",
    ),
    ExperimentSpec(
        "E10_esqs_yoy_only",
        signals=["ESQS_z"],
        weights={"ESQS_z": 1.0},
        notes="Isolation test: YoY-fixed ESQS vs original trailing-mean ESQS",
    ),
    # --- Phase 4: PEAD-based experiments ---
    ExperimentSpec(
        "E11_PEAD_only",
        signals=["PEAD_z"],
        weights={"PEAD_z": 1.0},
        hold_days=20,
        beta_neutral=True,
        notes="PEAD baseline: 1-day earnings reaction, 20d hold, beta-neutral",
    ),
    ExperimentSpec(
        "E12_PEAD_BSQ",
        signals=["PEAD_z"],
        weights={"PEAD_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        beta_neutral=True,
        notes="PEAD + BSQ balance sheet quality hard filter",
    ),
    ExperimentSpec(
        "E13_PEAD_IFR",
        signals=["PEAD_z", "IFR_z"],
        weights={"PEAD_z": 0.7, "IFR_z": 0.3},
        hold_days=20,
        beta_neutral=True,
        notes="PEAD + signed/accumulated IFR, beta-neutral",
    ),
    ExperimentSpec(
        "E14_PEAD_IFR_IVRS",
        signals=["PEAD_z", "IFR_z"],
        weights={"PEAD_z": 0.7, "IFR_z": 0.3},
        hold_days=20,
        ivrs_multiplier=True,
        regime_filter=True,
        beta_neutral=True,
        notes="PEAD + IFR + idiosyncratic IVRS multiplier + HMM regime gate",
    ),
    ExperimentSpec(
        "E15_PEAD_full_stack",
        signals=["PEAD_z", "IFR_z"],
        weights={"PEAD_z": 0.7, "IFR_z": 0.3},
        hold_days=20,
        bsq_filter=True,
        ivrs_multiplier=True,
        regime_filter=True,
        beta_neutral=True,
        notes="Full Phase 4 stack: PEAD + IFR + IVRS + BSQ + regime + beta-neutral",
    ),
    ExperimentSpec(
        "E16_BSQ_only",
        signals=["BSQ_z"],
        weights={"BSQ_z": 1.0},
        hold_days=20,
        bsq_filter=False,
        beta_neutral=True,
        notes="BSQ composite score as direction signal only (long quality / short junk); isolates quality factor from PEAD",
    ),
    # --- Phase 5: SUE + volatility targeting ---
    ExperimentSpec(
        "E17_SUE_BSQ_volTarget",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        beta_neutral=False,
        vol_target=0.15,
        notes="SUE (analyst consensus beat/miss) + BSQ filter + 15% vol target; requires paid consensus data",
    ),
    ExperimentSpec(
        "E18_full_with_SUE",
        signals=["SUE_z", "PEAD_z", "IFR_z"],
        weights={"SUE_z": 0.5, "PEAD_z": 0.35, "IFR_z": 0.15},
        hold_days=20,
        bsq_filter=True,
        ivrs_multiplier=True,
        regime_filter=True,
        beta_neutral=False,
        vol_target=0.15,
        notes="Full stack with paid SUE replacing ESQS; 15% vol target; best-case paid-data scenario",
    ),
    # --- Phase 6: Risk management + SUE signal quality improvements ---
    ExperimentSpec(
        "E19_vol_fix",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        notes="E17 + corrected L/S vol targeting formula (isolate MaxDD impact)",
    ),
    ExperimentSpec(
        "E20_vol_fix_stoploss",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E19 + 10% per-position stop-loss (isolate stop-loss impact)",
    ),
    ExperimentSpec(
        "E21_sue_normalizer",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E20 + rolling 4-quarter historical error std as SUE normalizer",
    ),
    ExperimentSpec(
        "E22_sue_magnitude",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E20 + SUE magnitude position weighting (|SUE_z| clipped [0.5, 3.0])",
    ),
    ExperimentSpec(
        "E23_phase1_full",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="All Phase 1 changes: vol fix + stop-loss + normalizer + magnitude weighting",
    ),
    ExperimentSpec(
        "E24_revision_dir",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        notes="E23 + EPS revision direction proxy filter",
    ),
    ExperimentSpec(
        "E25_revision_magnitude",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        notes="Full Phase 1+2 stack: E23 + revision direction + magnitude combined",
    ),
    ExperimentSpec(
        "E26_portfolio_vol_fix",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        concurrent_vol_adjust=True,
        rho_cross_cohort=0.0,
        notes="E25 + portfolio-level vol control: divide per-cohort target by sqrt(n_concurrent); rho=0 (over-corrects)",
    ),
    ExperimentSpec(
        "E27_portfolio_vol_fix_v2",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        concurrent_vol_adjust=True,
        rho_cross_cohort=-0.05,
        notes="E26 + corrected cross-cohort rho=-0.05; denom=sqrt(k(1+(k-1)rho)); targets 15% portfolio vol",
    ),
    ExperimentSpec(
        "E28_min_sue_threshold",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        concurrent_vol_adjust=True,
        rho_cross_cohort=-0.05,
        min_sue_z=0.5,
        notes="E27 + drop trades where |SUE_z| < 0.5 to filter near-zero signals and improve IC",
    ),
]


class AblationRunner:
    def __init__(self, experiments: Optional[list[ExperimentSpec]] = None):
        self.experiments = experiments if experiments is not None else ABLATION_MATRIX
        self.results: dict[str, dict] = {}

    def record(self, name: str, metrics: dict) -> None:
        self.results[name] = metrics

    def summary_table(self) -> list[dict]:
        return [
            {
                "experiment": exp.name,
                "signals": str(exp.signals),
                "hold_days": exp.hold_days,
                "regime_filter": exp.regime_filter,
                "ivrs_multiplier": exp.ivrs_multiplier,
                "bsq_filter": exp.bsq_filter,
                "beta_neutral": exp.beta_neutral,
                "vol_target": exp.vol_target,
                "stop_loss_pct": exp.stop_loss_pct,
                "use_revision_weight": exp.use_revision_weight,
                "concurrent_vol_adjust": exp.concurrent_vol_adjust,
                "rho_cross_cohort": exp.rho_cross_cohort,
                "min_sue_z": exp.min_sue_z,
                **self.results.get(exp.name, {}),
            }
            for exp in self.experiments
        ]

    def summary_df(self) -> pl.DataFrame:
        rows = self.summary_table()
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    def best_by(self, metric: str = "sharpe") -> Optional[ExperimentSpec]:
        best_exp = None
        best_val = float("-inf")
        for exp in self.experiments:
            val = self.results.get(exp.name, {}).get(metric, None)
            if val is not None and not (isinstance(val, float) and val != val):
                if val > best_val:
                    best_val = val
                    best_exp = exp
        return best_exp
