Etcd Capabilities
=================

Etcd integration is a powerful feature of Savant, allowing pipelines dynamically configure themselves based on the configuration stored in Etcd. This document describes the capabilities of Etcd integration in Savant.

About Etcd
----------

Etcd is a software designed specifically to dynamically configure software components. It supports fault-tolerant configuration (when it is required) allowing configuration clients operating even in the situation when certain parts of the system experiences outage.

Etcd exports HTTP-based API, easing system development and operation behind firewalls. It is also a secure system, Etcd supports username/password authentication, together with TLS-based authentication and encryption.

Etcd provides tree-like storage having the following key features:

- **Subscription**: clients automatically know about changes without the need to poll nodes for updates (watches);
- **Lease TTL**: certain nodes can be created with the TTL and notify subscribers automatically upon expiration;
- **Versioning**: clients can track node versions to determine changes.

In many cases, Etcd is a more simple way to implement configuration management rather than RESTful APIs, especially considering the above-discussed features. That is why we promote it for the project.

Etcd is beneficial to Savant as it enables watch-based configuration update. In computer vision pipelines, polling 3rd-party sources is not an option, as it is not efficient and may lead to unnecessary load. Etcd allows us to subscribe to changes and update the configuration in real-time.

Savant uses its own Etcd client implementation, free of GIL and supporting zero-latency access to monitored nodes. Software never waits for data to arrive, it is always available locally.

How to Use Etcd in Savant
--------------------------

To configure Etcd, users specify the following parameters in module configuration:

.. code-block:: yaml

    parameters:
      etcd:
        hosts: [etcd:2379]
        credentials:
          username: user
          password: password
        tls:
          ca: /path/to/ca.crt
          cert: /path/to/cert.crt
          key: /path/to/key.key
        watch_path: savant
        connect_timeout: 5
        watch_path_wait_timeout: 5

The meaning of the parameters is as follows:

- **hosts**: list of Etcd hosts to connect to;
- **credentials (optional)**: username and password to authenticate;
- **tls (optional)**: paths to CA, certificate, and key files;
- **watch_path**: path to watch for changes;
- **connect_timeout**: timeout for connection to Etcd;
- **watch_path_wait_timeout**: timeout for initial waiting for watched path to appear if it does not exist.

When you need accessing Etcd from the code, you can use the following API:

.. code-block:: python

    from savant_rs.utils import eval_expr

    CACHE_TTL = 60
    val, is_cached = eval_expr(f'etcd("path", "default_value")', ttl=CACHE_TTL, no_gil=True)

The function `eval_expr` is a part of the Savant runtime system and is used to evaluate expressions in the context of the Savant runtime system. The function `etcd` is a built-in function that allows you to access Etcd. The first argument is the path to the node in Etcd, and the second argument is the default value to return if the node does not exist; the type of the default value defines the type, a retrieved Etcd value will be cast if possible.

The third argument is the time-to-live for the cache, and the fourth argument is a flag indicating whether the function should be executed without GIL: you normally do not need to change it if you do not know what it is.

The function returns a tuple with the value and a boolean flag indicating whether the value is cached.

More information on Etcd usage in Savant can be found in a `conditional video processing sample <https://github.com/insight-platform/Savant/tree/develop/samples/conditional_video_processing>`__.