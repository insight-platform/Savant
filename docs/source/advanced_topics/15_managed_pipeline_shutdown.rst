Managed Pipeline Shutdown
--------------------------

There are two ways to make the pipeline complete properly with a 3rd-party component:

* with a shutdown message propagated by the upstream component;
* with the embedded webserver through REST API.

Shutdown Message
^^^^^^^^^^^^^^^^

The first approach requires that the 3rd-party component was able to send a shutdown message to the pipeline
ZMQ socket. A ready-to-use functionality is implemented in synchronous :py:class:`savant.client.runner.source.SourceRunner`
and asynchronous :py:class:`savant.client.runner.source.AsyncSourceRunner` classes.

The user must provide the shutdown token configured for a pipeline in the module parameters:

* ``parameters.shutdown_auth``: the token to be used for the shutdown message.

By default, this parameter is not defined and the shutdown message is not accepted.

REST API
^^^^^^^^

The second approach is to use the embedded webserver with REST API. The user can send a POST request
to the pipeline. The API supports two variants of the shutdown behavior:

* ``graceful``: the shutdown happens when the next ingress message is received, this approach lets ensure that the pipeline
  processes all inflight data;
* ``signal``: the shutdown happens immediately, the pipeline receives SIGINT or another signal configured with
  `savant-rs <https://insight-platform.github.io/savant-rs/modules/savant_rs/webserver.html#savant_rs.webserver.set_shutdown_signal>`__.

.. http:post:: /shutdown/(str: shutdown_auth)/(str: shutdown_type)

   :param shutdown_auth: the token to be used for the shutdown message
   :type shutdown_auth: str
   :param shutdown_type: the type of the shutdown message, one of ``graceful`` or ``signal``
   :type shutdown_type: str

   :status 200: when the shutdown message is accepted
   :status 401: when the shutdown message is not accepted
   :status 500: when the shutdown message is not configured\
