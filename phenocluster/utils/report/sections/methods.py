"""Methods section."""

from typing import Dict

from .._helpers import safe


def generate_methods_section(data: Dict) -> str:
    """Generate methods section."""
    config = data.get("config", {})
    model_cfg = config.get("model", {})
    stepmix = model_cfg.get("stepmix", {})
    model_sel = model_cfg.get("selection", {})
    data_split = config.get("data", {}).get("split", {})

    train_pct = int((1 - data_split.get("test_size", 0.2)) * 100)
    test_pct = int(data_split.get("test_size", 0.2) * 100)

    if model_sel.get("enabled", True):
        model_section = f"""
        <h3>Model Selection</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Selection Criterion</td>
            <td>{safe(model_sel.get("criterion", "BIC"))}</td></tr>
            <tr><td>Cluster Range Evaluated</td>
            <td>{safe(model_sel.get("min_clusters", 2))} - \
{safe(model_sel.get("max_clusters", 5))}</td></tr>
            <tr><td>Minimum Cluster Size</td>\
<td>{safe(model_sel.get("min_cluster_size", 30))}</td></tr>
            <tr><td>Random Initializations</td>\
<td>{safe(stepmix.get("n_init", 20))}</td></tr>
            <tr><td>Maximum Iterations</td>\
<td>{safe(stepmix.get("max_iter", 1000))}</td></tr>
        </table>"""
    else:
        n_clusters = model_cfg.get("n_clusters", "N/A")
        model_section = f"""
        <h3>Model Configuration</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Fixed Number of Clusters</td>\
<td>{safe(n_clusters)}</td></tr>
            <tr><td>Random Initializations</td>\
<td>{safe(stepmix.get("n_init", 20))}</td></tr>
            <tr><td>Maximum Iterations</td>\
<td>{safe(stepmix.get("max_iter", 1000))}</td></tr>
        </table>"""

    return f"""
    <section id="methods">
        <h2>Methods</h2>
        {model_section}

        <h3>Data Splitting</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Training Set</td><td>{train_pct}%</td></tr>
            <tr><td>Test Set</td><td>{test_pct}%</td></tr>
        </table>
    </section>
"""
