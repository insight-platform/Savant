The goal of testing framework to implement source -> blackbox -> sink to test the pipeline in the future. We must do that in a predictable way. Thus, we are going to send a static image and assess inverred results on the other end. We are going to use savant_rs library. The implementation (both source, blackbox and sink must use https://github.com/insight-platform/savant-rs/pkgs/container/savant-rs-py314/690736621?tag=savant-latest container).

The mock topology is:
source (dealer+connect) -> 
    (router+bind) infer_blackbox (dealer+bind) -> 
        (router+connect) sink

Prerequisites:
- You install Ultralytics YOLO in the container.
- Use logging from savant_rs.

What source does:
- socket is configured with the env.
- amount of repetitions with the env.
- use BlockingWriter;
- it works with a JPEG image provided externally.
- when it starts, it merges JPEG side-by-side horizontally (L|R) and saves it temporarily;
- when the service bootstraps it runs YOLOV11M for a JPEG and finds all person boxes.
- it saves those boxes into frame attribute (source, person) = [AttributeValue.integers(l, t, r, b), AttributeValue.float(conf), ..., ...]
- sends EOS finally (after all repetitions);
- stops;

What infer_blackbox does:
- sockets are configured with the env;
- use BlockingReader, BlockingWriter;
- creates top level ROIs for the left and right sides;
- receives frames;
- it runs yolo on incoming JPEG frames;
- it creates VideoObject instances for persons detected and attaches them to the ROIS as child objects;
- it sends frames to the sink;
- when EOS received, sends EOS to the sink and stops;

What sink does:
- socket is configured with the env;
- use BlockingReader;
- extracts objects and compare them with data saved in the (source, person) attribute with IoU metric (exists for BBox).
- if something does not match, - error is on console.
- if EOS received, stops.


Implementation in:
- Savant/samples/meta_merge/test
- Use ruff to format the code.
- Use typed implementation.

Work until the test is ready and passes with Docker compose (every service is presented by a separate Compose service: source, infer, sink). Finally provide a command for me to launch and test.

NOTE: if you find discrepancies in savant-rs API which need to be fixed, report them in the testing_framework_notes.md