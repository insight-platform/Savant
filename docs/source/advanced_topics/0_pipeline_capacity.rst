Pipeline Capacity
====================

Savant allows setting the pipeline capacity parameter to define how many streams a pipeline can simultaneously process. The parameter is defined with the ``max_parallel_streams`` property in the general parameters section. When the number of streams directed to the pipeline is higher than the value set by the parameter, the pipeline will terminate with the error.

This parameter is set to ``64`` by default, a pretty high value for the real-time pipeline. Consider this parameter as a trigger determining the cases when you wrongly directed more streams to the pipeline than it is expected to handle on particular hardware. Keep the parameter reasonably low for real-time pipelines to not break real-time constraints. In contrast, you may want to increase the parameter for non-real-time pipelines to ensure that the pipeline meets sharding expectations.

It is important to remember that in certain situations, the pipeline can technically experience more simultaneous streams than expected, e.g., when one stream is dead but not yet evicted, and another appears. Knowing that, set the parameter slightly higher to avoid pipeline termination in transition situations.