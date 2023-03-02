import napari


class TracingLayers:
    def __init__(self) -> None:
        self.terminal_points_layer: napari.layers.points.points.Points = None
        self.result_tracing_layer: napari.layers.points.points.Points = None
