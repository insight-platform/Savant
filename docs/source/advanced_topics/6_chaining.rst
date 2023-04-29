Pipeline Chaining
==================

With Savant, you can create chains of pipelines decomposing the processing based on a particular principle, e.g. distributing the processing between nodes or GPUs. Chaining is possible due to the communication protocol used by Savant: you can connect modules on the same node with the ``ipc://`` scheme or different nodes with the ``tcp://`` scheme.

To implement chaining, you attach the later module's source to the source module's sink. You may use all socket types like ``pub/sub``, ``dealer/router``, and ``req/rep`` to connect chain elements.

What reasons and arguments for implementing chaining rather than crafting a single module?
Let us provide you with several:

1. Distribute the workload in case a single GPU cannot carry out all the workload, or you want to use a grid of cheap GPUs to carry out commodity operations and a small number of expensive GPUs to carry out sophisticated operations if a cheap GPU discovers valuable information.
2. Distribute the processing between the edge and the core: on edge, you run primary perception operations and send the heavyweight processing to the data center; this scheme is beneficial as the core skips specific frames if there is no metadata required to process them; the edge also can avoid sending data for the frames which does not include valuable information.
3. Access to ready-to-use module you cannot change or incorporate in your pipeline, e.g., because of IP/license restrictions;
4. You would like to build a routed network of processing where data flow according to specific rules.

Efficiency
----------

Yes, it is the encoding and decoding are almost free on Nvidia GPUs, you usually don't care about additional operations like that.

Drawbacks
---------

The most significant drawback is the increased latency of a chain. Every decoding/encoding adds about 100 ms when processing a 30 FPS stream; so chaining multiplies the delay.