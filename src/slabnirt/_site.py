"""Shared site-identity policy for spatial workflows."""

SITE_COORDINATE_DECIMALS = 6


def site_key(x: float, y: float) -> tuple[float, float]:
    """Return the identity key of a survey site: (x, y) rounded to six decimals.

    The rounding only groups traces; plots and exports use the unrounded
    coordinates. It is a bucket, not a distance tolerance: points 1e-7 apart
    can fall into two sites, points almost 1e-6 apart into one.
    """
    return (
        round(float(x), SITE_COORDINATE_DECIMALS),
        round(float(y), SITE_COORDINATE_DECIMALS),
    )
