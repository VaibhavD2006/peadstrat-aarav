from dataclasses import dataclass, field

@dataclass
class ExperimentSpec:
    name: str
    signals: list[str]
    weights: dict[str, float]
    hold_days: int = 10
    notes: str = ""

class AblationRunner:
    def __init__(self, experiments: list[ExperimentSpec]):
        self.experiments = experiments
        self.results: dict[str, dict] = {}

    def record(self, name: str, metrics: dict) -> None:
        self.results[name] = metrics

    def summary_table(self) -> list[dict]:
        return [
            {"experiment": exp.name, "signals": str(exp.signals),
             **self.results.get(exp.name, {})}
            for exp in self.experiments
        ]
