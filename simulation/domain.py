"""Computational geometry is local SI coordinates, not a latitude modulo."""


def create_spatial_domain(experiment):
    return experiment.grid()


def latlon_to_local_xy(*args, **kwargs):
    raise NotImplementedError(
        "Geographic coordinates are metadata. A surveyed geometry and explicit projected CRS are required for geospatial simulation."
    )
