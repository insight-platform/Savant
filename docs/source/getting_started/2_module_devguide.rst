Module Development Guide
========================

Up until this point we expect you completed the steps:

- :doc:`0_configure_prod_env`;
- :doc:`1_configure_dev_env` for IDE of your choice.

This section focuses on the current best practices you can use when developing pipelines in Savant. We propose the process because we believe it is efficient, time-saving and brings the best user experience. We will highly appreciate if you share with us your experience and recommend improvements.

The section is in the beginning of the documentation to provide you with the introductory feeling; we understand that currently there are too many unknown concepts required to build the whole mindmap. It is absolutely a great idea to read carefully all "Savant 101" before practically developing pipelines.

Why Does It Matter?
-------------------

In software development, dealing with inefficient tooling can be immensely frustrating. Consider the situation where a developer needs to introduce a minor change to the codebase. On the surface, this change appears simple. However, once executed, the developer is forced to endure a long build and run process. Those extended minutes of waiting, magnified by anticipation, often culminate in disappointment when the task fails. The next steps involve sifting through extensive logs, pinpointing the issue, and once again enduring the tedious cycle of change, build, and run. Such iterative loops disrupt workflow, waste precious time, and hinder productivity, highlighting the crucial importance of efficient tooling and streamlined processes.

That is what vanilla DeepStream is about. Imagine a developer adjusting a detection code in a DeepStream application for vehicle tracking. After this minor change, they must restart the entire pipeline to test it. Due to the complexity of video analytics, it takes considerable time to build and launch the pipeline, and process the frames. Once it introduces a corner case, the pipeline crashes. The developer now sifts through dense GStreamer and DeepStream logs to diagnose the issue. This iterative process—tweak, wait, crash, diagnose—becomes a tedious cycle, highlighting the need for efficient debugging tools in intricate platforms like DeepStream.

When we started Savant, we had incapable, inefficient tooling as DeepStream has. But in Savant 0.2.5, we made many improvements to give developers efficient tooling, disrupting the annoying "change, build, run, read logs" cycle. We want Savant developers to have tooling enabling building integration tests and discovering problems in seconds, not minutes.

To help develop modules efficiently Savant provides provides two features and one auxiliary technology:

- **Client SDK**: a Python programmatic API allowing developers to send media and meta to Savant and receive the results right from Python programs without threads, complex logic, etc;

- **Dev Server**: a feature enabling pipeline reloading Python code automatically on change, significantly saving the developer's time working on Python code.

The above-mentioned auxiliary technology is `OpenTelemetry <https://opentelemetry.io/>`_. It is a broad topic, but for now, you may think that it helps profiling the code performance and collecting logs and events happening in the pipeline.

Let us observe basic operations you may encounter during the pipeline development and our approach to carry on them efficiently.

Recommended Module Layout
-------------------------

Small and medium-size pipelines can have a flat layout with all the components placed in a single directory. You can find a number of such pipelines in `samples <https://github.com/insight-platform/Savant/tree/develop/samples>`_ directory.

You don't need to have a whole Savant repo to develop a module, only what is inside a module directory.

The `my-module` created from `template <https://github.com/insight-platform/Savant/tree/develop/samples/template>`_ has more universal structure suitable for advanced pipelines with dependencies. Let us discuss the structure briefly:

- **.devcontainer**: the directory is required only if you are developing with VSCode; it contains Docker specification for VSCode. You may find two flavors there - for x86+dGPU and for Nvidia Jetson SOCs.
- **docker**: the directory contains two Dockerfiles containing stub code for building custom development Docker containers, installing additional requirements from the ``requirements.txt``; it is often a way to go if the pipeline requires Python libraries not delivered with Savant base images.
- **.dockerignore**: the file contains exclusions for files and paths typically must be avoided to add to a Docker image; you can extend them to reflect your actual needs, e.g. exclude media files.
- **Makefile** contains commands for building and launching the Docker image in a standalone mode, without IDE.
- **README.md** is just a README stub.
- **requirements.txt** contains additional Python requirements to add to the custom Docker image.
- **module** directory with actual pipeline code.

The `module` Directory
^^^^^^^^^^^^^^^^^^^^^^

The **module.yml** contains pipeline manifest describing pipeline stages and properties. This is where you the structure and declarative behavior with predefined properties and specify custom Python invocations when you need domain-specific processing. You can think of it as of ``HTML`` with ``JavaScript`` blocks if you have web-development experience.

There are three Python files ``run.py``, ``overlay.py``, ``custom_pyfunc.py``. Let us discuss them.

The  file ``run.py`` is an auxiliary file used as a module entrypoint. Docker images are configure to use it. Also, you use it to run and stop the pipeline from IDE. For production deployments, you normally don't need it, but you can use it to implement custom code initialization if necessary.

The file ``custom_pyfunc.py`` is a stub file for a `pyfunc` element which implements custom logic. You will find a lot of various pyfuncs in `samples <https://github.com/insight-platform/Savant/tree/develop/samples>`.

The file ``overlay.py`` represents a custom `draw_func` element used to customize video drawing functionality of Savant. It is usually used when you need to add some non-standard fancy graphics in your frames. The samples demonstrate custom ``draw_func`` implementations as well.

Rebuilding Docker Image
-----------------------

.. warning::

    Rebuilding the Docker image and reconfiguring IDE for a new one is a really time-killing operation. Plan the development process to avoid doing that frequently.

.. note::

    In this section we don't distinguish between ``Dockerfile.l4t`` and ``Dockerfile.x86``. Just keep in mind that when ``Dockerfile`` is mentioned, you must consider a variant, specific to your working environment.

Sometimes you may need to add components to Docker image. It could be a python requirement which you add to ``requirements.txt`` or a system utility specified immediately in ``Dockerfile`` build step. Whenever your changes affect ``Dockerfile`` you must rebuild it.

Rebuilding ``Dockerfile`` is done with:

.. code-block:: bash

    make build

After the rebuilding ``Dockerfile`` you must apply changes in your IDE.

Using The New Image In IDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: How to do.

Running The Module
------------------

If you carefully read the ":doc:`1_configure_dev_env`" section, you already know how to manage it.

In PyCharm:

.. image:: ../_static/img/dev-env/10-run-output-1.png

In VSCode:

.. image:: ../_static/img/dev-env/16-run-python-file.png

Also you can do it from the CLI like:

.. code-block:: bash

    python /opt/project/module/run.py

Use IDEs controls to stop the module as usual.

You must use hard restarts when introducing changes in the module’s YAML manifest. The YAML manifest corresponds to the pipeline; Savant does not implement rebuilding the pipeline on change.

These starts/stops are time-consuming; we recommend building the development process to decrease the number of such restarts. From our perspective, it can be achieved by development iteratively (from the pipeline beginning to the pipeline end) following the waterfall principle.

.. warning::

    We recommend avoiding the approach of defining the whole pipeline from scratch and debugging it end-to-end as a whole: it may be a very time-consuming and error-prone process. Define and troubleshoot pipeline stages one-by-one following the waterfall principle.

.. include:: /advanced_topics/9_dev_server.rst

.. include:: /advanced_topics/9_open_telemetry.rst

.. include:: /advanced_topics/10_client_sdk.rst


Using uri-input.py Script
-------------------------

Read from web cam, display with AO-RTSP, meta with Client SDK.

Summary
-------

TODO