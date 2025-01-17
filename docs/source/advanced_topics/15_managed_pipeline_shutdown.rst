Managed Pipeline Shutdown
--------------------------

There are two ways to make the pipeline complete properly with a 3rd-party component:

* with a shutdown message propagated by the upstream component;
* with the embedded webserver through REST API.

The first approach requires that the 3rd-party component was able to send a shutdown message to the pipeline
ZMQ socket. A ready-to-use functionality is implemented in synchronous :py:class:`savant.client.runner.source.SourceRunner`
and asynchronous :py:class:`savant.client.runner.source.AsyncSourceRunner` classes.

The user must provide the shutdown token configured for a pipeline in the module parameters:

* ``parameters.shutdown_auth``: the token to be used for the shutdown message.

By default, this parameter is not specified and the shutdown message is not accepted.

