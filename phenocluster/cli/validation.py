"""Configuration validation helpers used by the `validate-config` command."""

import typer

from .console import console


def _validate_structure(cfg, errors):
    """Check config structural validity (no data file needed)."""
    if not cfg.continuous_columns and not cfg.categorical_columns:
        errors.append("At least one of continuous_columns or categorical_columns must be non-empty")

    if cfg.outcome.enabled and not cfg.outcome.outcome_columns:
        errors.append("outcome.enabled=true but no outcome_columns specified")

    if not (0.0 < cfg.data_split.test_size < 1.0):
        errors.append(f"data.split.test_size must be in (0, 1), got {cfg.data_split.test_size}")

    if cfg.model_selection.enabled:
        if cfg.model_selection.min_clusters < 2:
            errors.append("model.selection.min_clusters must be >= 2")
        if cfg.model_selection.max_clusters < cfg.model_selection.min_clusters:
            errors.append("model.selection.max_clusters must be >= min_clusters")

    if cfg.survival.enabled and cfg.survival.targets:
        for t in cfg.survival.targets:
            if not t.time_column or not t.event_column:
                errors.append(
                    f"Survival target '{t.name}' must have both time_column and event_column"
                )

    _validate_multistate_structure(cfg, errors)

    if cfg.reference_phenotype.strategy not in ("largest", "healthiest", "specific"):
        errors.append(
            f"reference_phenotype.strategy must be 'largest', 'healthiest', "
            f"or 'specific', got '{cfg.reference_phenotype.strategy}'"
        )

    if cfg.feature_selection.enabled:
        if cfg.feature_selection.require_target and not cfg.feature_selection.target_column:
            errors.append(
                f"feature_selection.method='{cfg.feature_selection.method}' "
                f"requires a target_column"
            )


def _validate_multistate_structure(cfg, errors):
    """Validate multistate config structure."""
    if not cfg.multistate.enabled:
        return
    if not cfg.multistate.states:
        errors.append("multistate.states must be non-empty when enabled")
    if not cfg.multistate.transitions:
        errors.append("multistate.transitions must be non-empty when enabled")
    state_ids = {s.id for s in cfg.multistate.states} if cfg.multistate.states else set()
    if cfg.multistate.transitions:
        for tr in cfg.multistate.transitions:
            if tr.from_state not in state_ids:
                errors.append(
                    f"Transition '{tr.name}' references unknown from_state {tr.from_state}"
                )
            if tr.to_state not in state_ids:
                errors.append(f"Transition '{tr.name}' references unknown to_state {tr.to_state}")


def _check_columns(col_list, section_name, csv_columns, errors):
    """Flag config columns missing from the CSV."""
    for col in col_list:
        if col not in csv_columns:
            errors.append(f"{section_name}: column '{col}' not found in data")


def _validate_against_data(cfg, csv_columns, errors, warnings_list):
    """Cross-reference config column names against CSV header."""
    _check_columns(cfg.continuous_columns, "data.continuous_columns", csv_columns, errors)
    _check_columns(cfg.categorical_columns, "data.categorical_columns", csv_columns, errors)
    if cfg.outcome.enabled:
        _check_columns(cfg.outcome.outcome_columns, "outcome.outcome_columns", csv_columns, errors)

    if cfg.data_split.stratify_by:
        _check_columns([cfg.data_split.stratify_by], "data.split.stratify_by", csv_columns, errors)

    if cfg.feature_selection.enabled and cfg.feature_selection.target_column:
        _check_columns(
            [cfg.feature_selection.target_column],
            "preprocessing.feature_selection.target_column",
            csv_columns,
            errors,
        )

    if cfg.survival.enabled and cfg.survival.targets:
        for t in cfg.survival.targets:
            _check_columns(
                [t.time_column],
                f"survival.targets[{t.name}].time_column",
                csv_columns,
                errors,
            )
            _check_columns(
                [t.event_column],
                f"survival.targets[{t.name}].event_column",
                csv_columns,
                errors,
            )

    if cfg.multistate.enabled and cfg.multistate.states:
        for s in cfg.multistate.states:
            if s.event_column:
                _check_columns(
                    [s.event_column],
                    f"multistate.states[{s.name}].event_column",
                    csv_columns,
                    errors,
                )
            if s.time_column:
                _check_columns(
                    [s.time_column],
                    f"multistate.states[{s.name}].time_column",
                    csv_columns,
                    errors,
                )
        if cfg.multistate.baseline_confounders:
            _check_columns(
                cfg.multistate.baseline_confounders,
                "multistate.baseline_confounders",
                csv_columns,
                errors,
            )

    if cfg.reference_phenotype.strategy == "healthiest" and cfg.reference_phenotype.health_outcome:
        _check_columns(
            [cfg.reference_phenotype.health_outcome],
            "reference_phenotype.health_outcome",
            csv_columns,
            errors,
        )

    overlap = set(cfg.continuous_columns) & set(cfg.categorical_columns)
    if overlap:
        warnings_list.append(f"Columns in both continuous and categorical: {sorted(overlap)}")


def _display_validation_results(cfg, config, errors, warnings_list, csv_columns):
    """Display validation results to console."""
    if errors:
        console.print("[bold red]INVALID[/bold red] - config has errors:\n")
        for err in errors:
            console.print(f"  [red]x[/red] {err}")
    if warnings_list:
        if not errors:
            console.print("[bold yellow]WARNINGS:[/bold yellow]\n")
        else:
            console.print()
        for warn in warnings_list:
            console.print(f"  [yellow]![/yellow] {warn}")

    if not errors:
        console.print(f"\n[bold green]VALID[/bold green] - {config}")
        console.print(f"  Project: {cfg.project_name}")
        console.print(
            f"  Features: {len(cfg.continuous_columns)} continuous, "
            f"{len(cfg.categorical_columns)} categorical"
        )
        console.print(
            f"  Outcomes: {len(cfg.outcome.outcome_columns)}"
            f" ({'on' if cfg.outcome.enabled else 'off'})"
        )
        n_targets = len(cfg.survival.targets) if cfg.survival.targets else 0
        console.print(
            f"  Analyses: outcome={'on' if cfg.outcome.enabled else 'off'}, "
            f"survival={'on' if cfg.survival.enabled else 'off'} "
            f"({n_targets} targets), "
            f"multistate={'on' if cfg.multistate.enabled else 'off'}, "
            f"inference={'on' if cfg.inference.enabled else 'off'}"
        )
        if csv_columns is not None:
            console.print(f"  Data columns matched: all OK ({len(csv_columns)} in CSV)")
    else:
        raise typer.Exit(code=1)
