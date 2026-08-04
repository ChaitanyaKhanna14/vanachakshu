"""Turning a confirmed alert into something a person can act on.

A latitude and a longitude are not an alert. Someone has to decide whether to
drive out there, and they need to know where it is, how big it is, when it
happened, and how confident the system is.

Pure formatting — no network, no Earth Engine — so all of it runs in CI.

**Wording is a design constraint here, not a style preference.** Every message
says *possible forest disturbance, requires ground verification*. It never says
logging, encroachment, or illegal anything. A model output is not evidence of a
crime, and a system that phrases it as one can cause real harm to real people —
including people who were doing nothing wrong. The technical output is a change
in reflectance. What caused it is for a human to establish.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Final

from vanachakshu.alerts import TrackedAlert
from vanachakshu.config import AreaOfInterest

__all__ = [
    "GROUND_VERIFICATION_NOTICE",
    "alerts_to_geojson",
    "format_alert",
    "format_digest",
    "google_maps_url",
    "openstreetmap_url",
]

GROUND_VERIFICATION_NOTICE: Final = (
    "These are POSSIBLE forest disturbances detected from satellite imagery. "
    "They are not confirmed events and are not evidence of any offence. "
    "Every location requires ground verification before any action is taken."
)

# Zoom level that shows a half-hectare clearing together with enough surrounding
# ground to orient by - roads, ridgelines, settlements.
_MAP_ZOOM: Final = 16


def google_maps_url(lat: float, lon: float, zoom: int = _MAP_ZOOM) -> str:
    """Satellite view centred on the alert.

    Google's imagery is higher resolution than Sentinel-2, so this often
    resolves whether a detection looks like a real clearing before anyone
    travels. It is not current, though — usually months to years old — so it
    cannot confirm a *recent* event, only give context.
    """
    return f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},{zoom}z/data=!3m1!1e3"


def openstreetmap_url(lat: float, lon: float, zoom: int = _MAP_ZOOM) -> str:
    """OpenStreetMap view. Carries village names, tracks and forest boundaries,
    which is what someone planning a field visit actually needs."""
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map={zoom}/{lat:.6f}/{lon:.6f}"


def _size_description(area_ha: float) -> str:
    """Plain-language sense of scale.

    Hectares mean little to most readers. A comparison to something physical
    makes an alert legible to a journalist or a community member, not only to
    a forest officer.
    """
    if area_ha < 1.0:
        return f"about {area_ha * 10_000:,.0f} square metres"
    if area_ha < 10.0:
        return f"roughly {area_ha:.1f} football pitches"
    return f"about {area_ha:.0f} hectares"


def format_alert(alert: TrackedAlert, aoi: AreaOfInterest) -> str:
    """Render one alert as plain text.

    Plain text, not HTML or Rich markup, so the identical string works in a
    terminal, an email body, a Telegram message and a CI log.
    """
    return "\n".join(
        [
            f"Alert {alert.alert_id}  —  {aoi.name}",
            f"  Area          : {alert.area_ha:.2f} ha ({_size_description(alert.area_ha)})",
            f"  Location      : {alert.lat:.5f}, {alert.lon:.5f}",
            f"  First seen    : {alert.first_seen}",
            f"  Last seen     : {alert.last_seen}",
            f"  Confirmations : {alert.confirmations} separate satellite "
            f"{'pass' if alert.confirmations == 1 else 'passes'}",
            f"  Satellite view: {google_maps_url(alert.lat, alert.lon)}",
            f"  Map / names   : {openstreetmap_url(alert.lat, alert.lon)}",
        ]
    )


def format_digest(
    alerts: Sequence[TrackedAlert],
    aoi: AreaOfInterest,
    issued_on: date,
) -> str:
    """Render a batch of newly confirmed alerts as one message.

    Sorted largest first: with limited time, the biggest clearing is the one
    worth visiting. Sorting by detection order would bury it.
    """
    if not alerts:
        return (
            f"{aoi.name} — {issued_on.isoformat()}\n\n"
            "No newly confirmed forest disturbances in this cycle."
        )

    ordered = sorted(alerts, key=lambda a: a.area_ha, reverse=True)
    total_ha = sum(a.area_ha for a in ordered)

    header = [
        f"{aoi.name} — {issued_on.isoformat()}",
        "",
        f"{len(ordered)} newly confirmed possible disturbance"
        f"{'s' if len(ordered) != 1 else ''}, {total_ha:.2f} ha in total.",
        "",
        GROUND_VERIFICATION_NOTICE,
        "",
        "-" * 68,
        "",
    ]
    body = [format_alert(alert, aoi) for alert in ordered]
    return "\n".join(header) + "\n\n".join(body) + "\n"


def alerts_to_geojson(alerts: Sequence[TrackedAlert], aoi: AreaOfInterest) -> dict[str, Any]:
    """Export alerts as GeoJSON, loadable by a field GPS or QGIS.

    Points rather than the original polygons: the store keeps a centroid, and
    inventing a polygon shape from it would imply a precision that is not
    there. A point plus an area figure is honest about what is known.
    """
    return {
        "type": "FeatureCollection",
        "properties": {
            "aoi": aoi.name,
            "notice": GROUND_VERIFICATION_NOTICE,
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [alert.lon, alert.lat]},
                "properties": {
                    "alert_id": alert.alert_id,
                    "area_ha": round(alert.area_ha, 3),
                    "first_seen": alert.first_seen,
                    "last_seen": alert.last_seen,
                    "confirmations": alert.confirmations,
                    "notified_on": alert.notified_on,
                    "status": "possible disturbance - requires ground verification",
                },
            }
            for alert in sorted(alerts, key=lambda a: a.area_ha, reverse=True)
        ],
    }
