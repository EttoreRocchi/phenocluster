"""Unit tests for CohortReport / GeneralizabilityReport serialization."""

import json

import pandas as pd

from phenocluster.evaluation.generalizability import (
    CohortReport,
    GeneralizabilityReport,
)


def _make_report(label="window1", kind="temporal"):
    drift = pd.DataFrame(
        {
            "feature": ["x", "y"],
            "kind": ["continuous", "continuous"],
            "psi": [0.05, 0.5],
        }
    )
    return CohortReport(
        label=label,
        kind=kind,
        n_samples=120,
        cluster_distribution={0: {"count": 60, "percentage": 50.0}},
        log_likelihood=-100.0,
        drift=drift,
    )


def test_cohort_report_to_dict_keeps_dataframe():
    rep = _make_report()
    d = rep.to_dict()
    assert isinstance(d["drift"], pd.DataFrame)
    assert d["label"] == "window1"


def test_cohort_report_to_json_safe_records():
    rep = _make_report()
    safe = rep.to_json_safe()
    assert isinstance(safe["drift"], list)
    json.dumps(safe)


def test_generalizability_report_groups():
    t = _make_report("w1", "temporal")
    s = _make_report("siteA", "site")
    rep = GeneralizabilityReport(temporal=[t], multisite=[s], summary={"foo": "bar"})
    safe = rep.to_json_safe()
    assert len(safe["temporal"]) == 1
    assert len(safe["multisite"]) == 1
    assert safe["summary"] == {"foo": "bar"}
