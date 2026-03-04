"""Descriptive feature characterization section."""

from typing import Dict

from .._helpers import format_value, safe


def generate_feature_importance_section(data: Dict) -> str:
    """Generate feature importance section."""
    feature_imp = data.get("feature_importance", {})

    if not feature_imp:
        return """
    <section id="features">
        <h2>Descriptive Feature Characterization</h2>
        <p>Feature importance analysis not available.</p>
    </section>
"""

    top_features = feature_imp.get("top_features_per_cluster", {})

    has_groups = False
    for _cid, feats in top_features.items():
        if feats and isinstance(feats[0], dict) and "group" in feats[0]:
            has_groups = True
        break

    feature_tables = []
    for cluster_id, features in sorted(
        top_features.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0,
    ):
        rows = _build_feature_rows(features, has_groups)
        if rows:
            if has_groups:
                header = (
                    "<tr><th>Group</th><th>Feature</th>"
                    "<th>Effect Size</th><th>Type</th>"
                    "<th>Detail</th></tr>"
                )
            else:
                header = "<tr><th>Feature</th><th>Effect Size</th><th>Type</th><th>Detail</th></tr>"
            table_html = (
                f"<h3>Phenotype {safe(cluster_id)}"
                f" - Distinguishing Features</h3>\n"
                f"<table>\n{header}\n{''.join(rows)}\n</table>"
            )
            feature_tables.append(table_html)

    return f"""
    <section id="features">
        <h2>Descriptive Feature Characterization</h2>

        {
        "".join(feature_tables)
        if feature_tables
        else ("<p>No feature importance data available.</p>")
    }
    </section>
"""


def _build_feature_rows(features: list, has_groups: bool) -> list:
    """Build feature table rows."""
    rows = []
    for feat in features:
        name = feat.get("feature", feat.get("name", "Unknown"))
        importance = feat.get("importance", feat.get("effect_size", 0))
        feat_type = feat.get("type", "unknown")
        group = feat.get("group", "")

        if feat_type == "categorical":
            dom_cat = feat.get("dominant_category", "")
            dom_ratio = feat.get("dominant_ratio", 0)
            detail = f"{safe(dom_cat)} ({dom_ratio:.1f}x)" if dom_cat else ""
        else:
            direction = feat.get("direction", "")
            effect_d = feat.get("effect_size", 0)
            detail = f"{safe(direction)} (d={effect_d:.2f})" if direction else ""

        if has_groups:
            rows.append(
                f"<tr><td>{safe(group)}</td><td>{safe(name)}</td>"
                f"<td>{format_value(importance)}</td>"
                f"<td>{safe(feat_type)}</td><td>{detail}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{safe(name)}</td>"
                f"<td>{format_value(importance)}</td>"
                f"<td>{safe(feat_type)}</td><td>{detail}</td></tr>"
            )

    return rows
