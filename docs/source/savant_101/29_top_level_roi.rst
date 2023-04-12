Top-Level ROI
=============

In Savant models always work on a specified region of interest (ROI). By default, the framework automatically creates the ROI with the label ``frame`` corresponding to the whole frame. The models within the pipeline can work on either specific ROI labels defined on model input or on default ROI if no label is specified.

Model Units with no ROI specified may be treated as the primary model. However, it is just an agreement: sometimes, you may want to remove ``frame`` for a certain frame or a whole stream. E.g., if this stream was not configured, you definitely do not want to spend GPU resources to process its frames.

Another case is when you can delete the ``frame`` ROI conditionally to customize the interval of operations based on some heuristics. E.g., you may want to run the YOLO model once in a second while it does not detect objects. When it finds something, you may want to run for every frame until the situation changes.
