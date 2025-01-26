Embedded Key-Value Store
--------------------------

Savant pipeline provides an embedded key-value store available through:

* ABI for internal access (Python or Rust code);
* REST API for external access.

REST API is intended for 3rd-party components and downstream elements to exchange data with the pipeline. As the API
operates over HTTP and network it may introduce latency and should be used only for non-blocking operations.

The pipeline components which use KVS must utilize ABI to avoid the network overhead and communicate with it in a
deterministic and no-latency way.

The key-value store does not replace such solutions as Redis and is intended for small data exchange between pipeline
components. Storing several hundreds or even several thousand of keys is just fine.

However, the amount actually is determined by the way one uses the KVS: the more glob search operations are performed, the less keys can be stored.
When, KVS is used to operate only on keys with exact names, the amount of keys can be almost arbitrary.

KVS has the following properties:

* every key consists of (namespace, key) pair;
* every key has a value represented by :py:class:`savant_rs.primitives.Attribute`;
* an entry can have expiration TTL or be stored indefinitely;
* user can operate with exact keys or use glob search patterns;
* external API works on protocol buffers. Those, who use non-Rust/Python languages, can generate the client code from the protocol buffer `definition <https://github.com/insight-platform/savant-protobuf/blob/main/src/savant_rs.proto#L155>`__.
* external watch API provides JSON messages (and optionally protobuf-serialized :py:class:`savant_rs.primitives.AttributeSet`) via Websocket protocol for Set and Delete events; TTL-induced expiration events are not tracked;
* internal watch API provides a ready-to-use Python objects containing corresponding list of :py:class:`savant_rs.primitives.Attribute` for Set and Delete events; TTL-induced expiration events are not tracked.

Examples of Python-based API/ABI can be found `here <https://github.com/insight-platform/savant-rs/blob/main/python/webserver_kvs.py>`__.

Examples of Rust-based API/ABI can be found in ``savant-rs`` test suite:

* `ABI <https://github.com/insight-platform/savant-rs/blob/main/savant_core/src/webserver/kvs.rs>`__;
* `API <https://github.com/insight-platform/savant-rs/blob/main/savant_core/src/webserver.rs#L329-L482>`__.

Endpoints
^^^^^^^^^

Setting Keys
~~~~~~~~~~~~

.. http:post:: /kvs/set

    :form body: list of :py:class:`savant_rs.primitives.Attribute` serialized to protocol buffer :py:class:`savant_rs.primitives.AttributeSet` message
    :status 200: when the keys are set
    :status 400: when the keys are not set due to an deserialization error

Setting Keys with TTL
~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /kvs/set-with-ttl/(int: ttl)

    :param ttl: time to live in milliseconds (non-negative)
    :form body: list of :py:class:`savant_rs.primitives.Attribute` serialized to protocol buffer :py:class:`savant_rs.primitives.AttributeSet` message
    :status 200: when the keys are set
    :status 400: when the keys are not set due to an deserialization error

Deleting Keys by Glob
~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /kvs/delete/(str: namespace)/(str: name)

    :param namespace: glob pattern to match the namespace
    :param name: glob pattern to match the name
    :status 200: if the request is correct

Deleting Keys by Exact Match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /kvs/delete-single/(str: namespace)/(str: name)

    :param namespace: exact namespace
    :param name: exact name
    :status 200: if the request is correct

Getting Keys Names by Glob
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /kvs/search-keys/(str: namespace)/(str: name)

    :param namespace: glob pattern to match the namespace
    :param name: glob pattern to match the name
    :status 200: if the request is correct

    **Response**

    .. sourcecode:: http

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
            ["namespace1", "key1"],
            ["namespace2", "key2"],
            ["namespace3", "key3"]
        ]

Getting Attributes by Glob
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /kvs/search/(str: namespace)/(str: name)

    :param namespace: glob pattern to match the namespace
    :param name: glob pattern to match the name
    :status 200: if the request is correct

    **Response**

    .. sourcecode:: http

        HTTP/1.1 200 OK

        serialized AttributeSet

Getting Attributes by Exact Match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /kvs/get/(str: namespace)/(str: name)

    :param namespace: exact namespace
    :param name: exact name
    :status 200: if the request is correct

    **Response**

    .. sourcecode:: http

        HTTP/1.1 200 OK

        serialized AttributeSet

Watching for updates With Websocket Subscription
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To receive only key names and operation types, use the following endpoint:

.. code-block:: bash

    websocat -U --ping-interval 1 --ping-timeout 2 ws://localhost:8080/kvs/events/meta


Returns messages in JSON format:


.. code-block:: json

    [{
        "name":"frame_counter",
        "namespace":"counter",
        "operation":"set",
        "timestamp": {
            "nanos_since_epoch": 825169275,
            "secs_since_epoch": 1737820251
        },
        "ttl":null
    }]


To receive key names, operation types and serialized attributes, use the following endpoint:

.. code-block:: bash

    websocat -U --ping-interval 1 --ping-timeout 2 ws://localhost:8080/kvs/events/full


Returns messages in pairs (JSON - for metadata, serialized AttributeSet - for attributes):


.. code-block:: json

    [{
        "name":"frame_counter",
        "namespace":"counter",
        "operation":"set",
        "timestamp": {
            "nanos_since_epoch": 825169275,
            "secs_since_epoch": 1737820251
        },
        "ttl":null
    }]

.. code-block::

    <protobuf-serialized AttributeSet>
    ...


.. note::

    Take a look at Python-based `API/ABI <https://github.com/insight-platform/savant-rs/blob/main/python/webserver_kvs.py>`__ to learn how to decode the serialized :py:class:`savant_rs.primitives.AttributeSet`.
