import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Plotting
# ============================================================

def plot_observables(
    times,
    observables,
    *,
    output_path=None,
    show=True,
    columns=2,
    figsize=None,
):
    """Plot one or more groups of time-dependent observables.

    ``observables`` maps subplot titles to dictionaries containing a
    ``values`` array. Values may have shape ``(n_times,)`` or
    ``(n_curves, n_times)``. Optional entries are ``labels``, ``ylabel``,
    ``styles`` (one dictionary of Matplotlib options per curve), and
    ``zero_line``.
    """
    times = np.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    if not observables:
        raise ValueError("observables must contain at least one entry")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    rows = int(np.ceil(len(observables) / columns))
    if figsize is None:
        figsize = (7 * columns, 5 * rows)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for axis, (title, specification) in zip(flat_axes, observables.items()):
        values = np.asarray(specification["values"])
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != times.size:
            raise ValueError(
                f"{title!r} values must have shape (n_times,) or "
                "(n_curves, n_times)"
            )

        labels = specification.get("labels")
        if labels is None:
            labels = [None] * values.shape[0]
        if len(labels) != values.shape[0]:
            raise ValueError(f"{title!r} must have one label per curve")

        styles = specification.get("styles")
        if styles is None:
            styles = [{} for _ in range(values.shape[0])]
        if len(styles) != values.shape[0]:
            raise ValueError(f"{title!r} must have one style per curve")

        for curve, label, style in zip(values, labels, styles):
            axis.plot(times, curve, label=label, **style)

        if specification.get("zero_line", False):
            axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel(specification.get("ylabel", ""))
        if any(label is not None for label in labels):
            axis.legend(ncol=specification.get("legend_columns", 1))
        axis.grid(alpha=0.3)

    for axis in flat_axes[len(observables):]:
        axis.set_visible(False)

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes