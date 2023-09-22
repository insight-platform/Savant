Python Multithreading in Savant
===============================

Savant ``0.2.5`` introduced a new feature: multithreaded Python functions. By default, GStreamer is a multithreaded framework; however, Python has its own limitations regarding how multithreading is implemented.

The limitations are connected with the infamous GIL: a global mutex required to acquire when accessing Python data structures. A lot about GIL influence is written on `PythonSpeed <https://pythonspeed.com/articles/python-gil/>`__. To sum up: regular compute-related Python code cannot benefit from multithreading because GIL allows running only one thread which works with Python data structure. However, frameworks like NumPy or OpenCV release the GIL lock when running their algorithms, allowing Python to run in true parallel.

So, by enabling multithreaded execution in Savant, you may benefit from frameworks releasing GIL. However, you need to remember that when you release GIL, you must wait for it back after the release ends. Specifically, short-running operations suffer from excessive GIL releases.
Regular GIL-keeping Python operations looks like as follows:

.. code-block::

    op:
      Acquire_GIL  # takes time to wait when GIL is unlocked
        do_something
      Release_GIL  # free of charge

The operation releasing GIL looks like as follows:

.. code-block::

    op:
      Acquire_GIL # takes time to wait when GIL is unlocked
        Release_GIL
          do_something # free of charge
        Acquire_GIL # takes time to wait when GIL is unlocked
      Release_GIL # free of charge

You can see that the releasing operation waits for GIL twice, which is harmful in two situations of high GIL contention:

- threads with short-running GIL-free operations;
- threads with long-running GIL-keeping operations.

Remember, that when Python runs a single-threaded program, GIL contention is low and predictable (free of charge), when it runs a multi-threaded program, GIL contention must be carefully analyzed.

Rules of Thumb for Savant
-------------------------

1. Always start with single-threaded implementation and switch to multithreaded only when the pipeline benefits from it.

2. If a pipeline executes compute-intensive long-running NumPy or OpenCV operations when you implemented parts of your code in FFI C/Rust/Cython) and it releases GIL, the pipeline may benefit from multithreaded execution.

3. Don't release GIL in short-running operations (I would say less than 10 microseconds).

4. If a pipeline carries out I/O operations (databases, files) it can benefit from multithreading as they release GIL.

5. If CPU frequency is high single-threaded execution may give better results, if CPU frequency is low multithreaded execution may give better results.

How To Enable Multithreading in Savant
--------------------------------------

.. note::

    As for ``0.2.5`` Savant doesn't enable mandatory multithreading for a pipeline, you need to do it by yourself. In the future, this behavior may change if we find that most pipelines benefit from multithreading.

Python multithreading can be enabled by placing GStreamer ``queue`` elements before ``pyfunc`` and ``draw_func`` elements. So you need to enable those queues in YAML:

.. code-block:: yaml

    # disabled
    #
    parameters:
      buffer_queues: null

    # enabled
    #
    parameters:
      buffer_queues:
        length: 1        # keep in mind that every buffered frame occupies GPU
        byte_size: 0     # we don't recommend setting byte_size to specific values other than 0

.. warning::

    You may want setting ``length`` to a larger number to endure traffic bursts, but remember that every buffer has an associated raw frame in GPU, so, e.g. two ``pyfuncs``, one ``draw_func`` with ``length`` set to ``10`` and 16 of 720p sources processed may result in ``3 x 10 x 16 x 1280 x 720 x 4 (RGBA)`` which is almost ``1.7`` GB of GPU RAM.