
PyCharm Professional
====================

The official documentation on configuring a `docker <https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html>`_  interpreter is pretty complete, but there are caveats. Below, you will find an improved configuration guide helping you to set up the runtime quickly.

The guide tested with PyCharm 2023.1.4 Professional.

.. note::

    If you have a different version of PyCharm, we cannot guarantee that the settings will work and match the screenshots.

Project Preparation
-------------------

#. Clone the repo:

    .. code-block:: bash

        git clone https://github.com/insight-platform/Savant.git

#. Copy and rename the template (let's name the new project ``my-module``):

    .. code-block:: bash

        cp -r Savant/samples/template my-module

#. Start the IDE and create a new project in the ``my-module`` directory.

#. When opening the project, skip the step related to the environment creation: click **Cancel** on **Creating Virtual Environment**:

.. image:: ../_static/img/dev-env/01-cancel-creating-virtual-environment.png

Setting Up The Interpreter
--------------------------

#. Open the project settings with **File > Settings ...** or **Ctrl+Alt+S**.

#. Choose **Project: my-module > Python Interpreter** section.

#. Add a new **Python Interpreter** by clicking the **Add Interpreter**, and choosing **On Docker...**:

    .. image:: ../_static/img/dev-env/02-setup-python-interpreter.png

#. In the modal window:

  * **[Step 1 of 3]**: Choose Dockerfile: ``docker/Dockerfile.x86`` or ``docker/Dockerfile.l4t``, if you are on Jetson, and set up Context folder (should be set to the root of the project). In the **Optional** section you have to specify an **Image tag**:

    .. warning::
        If you do not specify **Image tag**, docker images and container will not update correctly.

    .. image:: ../_static/img/dev-env/03-setup-docker-build.png

  * **[Step 2 of 3]**: Make sure docker build is successful.

  * **[Step 3 of 3]**: Keep the default python interpreter (python3) and click **Create**:

    .. image:: ../_static/img/dev-env/04-setup-interpreter-in-docker.png

4. Make sure that the ``savant`` package is in the package list, click **OK**:

    .. image:: ../_static/img/dev-env/05-check-savant-package.png

#. Open the ``my-module/run.py`` file in the IDE.

#. Check that IDE `sees` dependencies pointing the mouse over the line with the ``import`` directive: the popup must appear with the description of the function:

    .. image:: ../_static/img/dev-env/06-check-func-spec.png

Configure Module Run
--------------------

The ``run.py`` file is the entrypoint of the module, let's configure the launch command for the script.

#. Click on the **Run** icon and choose **Modify Run Configuration**:

    .. image:: ../_static/img/dev-env/07-open-run-config.png

#. In the **Environment** section add ``PYTHONPATH=/opt/savant`` (PyCharm `rewrites <https://youtrack.jetbrains.com/issue/PY-32618/The-original-PYTHONPATH-is-replaced-by-PyCharm-when-running-configurations-using-Docker-interpreter>`_  ``PYTHONPATH``):

    .. image:: ../_static/img/dev-env/08-run-config-env.png

#. In the **Docker container settings** section add ``--gpus all``:

    .. image:: ../_static/img/dev-env/09-run-config-ops.png

#. Launch the module by choosing **Run 'run'** or with the hotkey **Ctrl+Shift+F10**:

    .. image:: ../_static/img/dev-env/10-run-output-1.png

#. You may see various GStreamer error messages: it's ok. At the end you will see the output with metadata:

    .. image:: ../_static/img/dev-env/11-run-output-2.png

That's it, the environment is set up. Now you are ready to develop your own pipeline. See the next section to find out how.

Update Runtime On Container Change
----------------------------------

.. include:: ../includes/getting_started/1_pycharm_update_docker.rst

In the following sections, you will find additional details on module development.
