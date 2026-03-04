"""Executive summary section."""

from typing import Dict

from .._helpers import safe


def generate_summary_section(data: Dict) -> str:
    """Generate executive summary section."""
    config = data.get("config", {})
    split_info = data.get("split_info", {})
    cluster_stats = data.get("cluster_stats", {})

    n_clusters = (
        len(cluster_stats) if cluster_stats else config.get("model", {}).get("n_clusters", "N/A")
    )
    n_train = split_info.get("train_size", "N/A")
    n_test = split_info.get("test_size", "N/A")
    data_cfg = config.get("data", {})
    n_continuous = len(data_cfg.get("continuous_columns", []))
    n_categorical = len(data_cfg.get("categorical_columns", []))

    model_sel = data.get("model_selection", {})
    best_criterion = model_sel.get(
        "criterion_used",
        model_sel.get(
            "criterion", config.get("model", {}).get("selection", {}).get("criterion", "BIC")
        ),
    )

    return f"""
    <section id="summary">
        <h2>Executive Summary</h2>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">{safe(n_clusters)}</div>
                <div class="label">Phenotypes Identified</div>
            </div>
            <div class="metric-card">
                <div class="value">{safe(n_train)}</div>
                <div class="label">Training Samples</div>
            </div>
            <div class="metric-card">
                <div class="value">{safe(n_test)}</div>
                <div class="label">Test Samples</div>
            </div>
            <div class="metric-card">
                <div class="value">{n_continuous + n_categorical}</div>
                <div class="label">Features Analyzed</div>
            </div>
        </div>

        <div class="summary-box">
            <strong>Key Findings:</strong> Latent class analysis identified
            {safe(n_clusters)} distinct phenotypes
            using {safe(best_criterion)} as the selection criterion.
            The model was trained on {safe(n_train)} samples
            and validated on a held-out test set of {safe(n_test)} samples.
        </div>
    </section>
"""
