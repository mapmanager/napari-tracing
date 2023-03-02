# Napari tracing plugin

## Workflow

1. Selecting an image layer for tracing
Create a combo box which shows a list of available image layers for which we can perform some tracing

- Have a button underneath it which says, choose image layer for tracing
- When a user chooses this layer,
    -set `self.image_layer` to this layer
    -create a `terminal points layer` for drawing this tracing;
        -set `self.terminal_points_layer` to this layer

1. Setting up global dictionaries and the current tracing object model

- Create a global dictionary which is a `layer_tracing_mapping` that maps each `layer_id` to a `list of tracings` done on that layer.
- create a `trace object` of the type `Trace Model` . Set its `layer_ID` field to current layer’s hash value. Its like a primary key.
- set its start and end points to the global `self.start_point` and the global `self.end_point` which the user selects.
- Upon successful tracing after getting a tracing result, save this result into `trace object` attribute `self.tracing_result`

1. Saving Tracing
When the user hits save create a `segment ID` for this `trace object` and save it in the `trace object` property called `self.segment_ID`
    - Have a global dictionary called `layer_segment_mapping` that maps each `layer_id` to its most recent element’s `segment_id`.
    - So when you have to set the current tracing’s segment ID, use the layer’s most current `segment ID` + 1.
    - update the `layer_segment_mapping` dictionary value for key `layer_id` to the new segment ID.
    - To query this `layer_segment_mapping` using the `layer_ID` property in the `trace_object` as key.
    - When the user hits `save trace`, append the `trace object model` to the current layer’s list of tracings in the `layer_tracing_mapping` .
    - open the appropriate CSV file and using the values in `tracing object` populate the layer rows.
    - Set the `trace object model, self.start_point` and `self.end_point` to None.
