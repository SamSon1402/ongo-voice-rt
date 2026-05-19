"""Turn-level telemetry aggregator.

Keeps a rolling window of TurnMetrics and exposes computed values
(avg TTFT, edge-resolved %, cost projection). Designed to be pulled
periodically by the dashboard.
"""

from ongovoice.telemetry.aggregator import TurnAggregator

__all__ = ["TurnAggregator"]
