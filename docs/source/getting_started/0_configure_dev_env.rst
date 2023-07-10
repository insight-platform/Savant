Configure The Development Environment
=====================================

The Savant framework requires a complex environment, including GStreamer, Nvidia DeepStream, OpenCV CUDA, and other dependencies. Installing and configuring all the required packages is a daunting task. Therefore, we prepared ready-to-use docker images both for development and production usage. The image provides you with a fully configured environment in minutes.

Popular Python IDEs like PyCharm Professional and VSCode support the feature to set up a development environment inside a  docker container: it gives the best opportunity to configure the dev runtime for Savant quickly. The Savant repository includes a module template for a quick start; we will use it to show you how to set up the development environment in popular IDEs.

.. note:: Unfortunately, the PyCharm Community edition doesn't support the environment on Docker, so you cannot use it to reproduce the presented instructions.

.. note:: The recommended spare space in a filesystem where docker images are stored is **20 GB**.

PyCharm Professional
--------------------

You can rely on the official documentation on configuring a `docker <https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html>`_  interpreter, but there are caveats. Below, is a modified configuration guide with screenshots and comments helping you to get started quickly.

The instruction is created and tested in PyCharm 2023.1 **Professional**.

Project Preparation
^^^^^^^^^^^^^^^^^^^

#. Clone the Savant repo:

    .. code-block:: bash

        git clone https://github.com/insight-platform/Savant.git

#. Copy and rename the template (let's name the new project ``my-module``):

    .. code-block:: bash

        cp -r Savant/samples/template my-module

#. Run the IDE and open a new project in the ``my-module`` directory.

#. When opening the project, skip the step related to the environment creation: click **Cancel** on **Creating Virtual Environment**:

.. image:: ../_static/img/dev-env/01-cancel-creating-virtual-environment.png

Setting Up The Interpreter
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Open the project settings with **File > Settings ...** or **Ctrl+Alt+S**.

#. Choose **Project: my-module > Python Interpreter** section.

#. Add a new **Python Interpreter** by clicking the **Add Interpreter**, and choosing **On Docker...**:

    .. image:: ../_static/img/dev-env/02-setup-python-interpreter.png

#. In the modal window:

  * **[Step 1 of 3]**: Choose Dockerfile: ``docker/Dockerfile.x86`` or ``docker/Dockerfile.l4t``, if you are on Jetson, and set up Context folder (should be set to the root of the project). In the **Optional** section you can specify an **Image tag**:

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
^^^^^^^^^^^^^^^^^^^^

The ``run.py`` file is the entrypoint of the module, let's configure the launch command for the script.

#. Click on the **Run** icon and choose **Modify Run Configuration**:

    .. image:: ../_static/img/dev-env/07-open-run-config.png

#. In the **Environment** section add ``PYTHONPATH=/opt/savant`` (PyCharm `rewrites <https://youtrack.jetbrains.com/issue/PY-32618/The-original-PYTHONPATH-is-replaced-by-PyCharm-when-running-configurations-using-Docker-interpreter>`_  ``PYTHONPATH``):

    .. image:: ../_static/img/dev-env/08-run-config-env.png

#. In the **Docker container settings** section add ``--gpus all``:

    .. image:: ../_static/img/dev-env/09-run-config-ops.png

#. Launch the module by choosing **Run 'run'** or with the hotkey **Ctrl+Shift+F10**:

    .. image:: ../_static/img/dev-env/10-run-output-1.png

#. You may see various GStreamer error messages: it's ok. At the end you will see pipeline's output with metadata:

    .. image:: ../_static/img/dev-env/11-run-output-2.png

That's it, the environment is set up. Now you are ready to develop your own pipeline: modify the module config (``module/module.yaml``), add your own components, etc.

Notes
^^^^^

PyCharm does not automatically detect newly installed packages in a Docker container. However, there is an option to manually scan for new packages: go to the **Settings** and look for **Rescan**, then navigate to **Plugins > Python > Rescan Available Python Modules and Packages** and set the hotkey (e.g., **Alt+R**):

.. image:: ../_static/img/dev-env/12-rescan.png

After adding a new package to the ``requirements.txt``, simply press the specified hotkey to rebuild the image and update the packages.

VS Code
-------

IDE Preparation
^^^^^^^^^^^^^^^

#. Install the `Remote Development <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_ extension pack.

Project Preparation
^^^^^^^^^^^^^^^^^^^

#. Clone the Savant repo:

    .. code-block:: bash

        git clone https://github.com/insight-platform/Savant.git

#. Copy and rename the template (let's name the new project ``my-module``):

    .. code-block:: bash

        cp -r Savant/samples/template my-module

#. Run the IDE and open the ``my-module`` folder.

Reopen in container
^^^^^^^^^^^^^^^^^^^

#. Open the command palette with **F1** or **Ctrl+Shift+P**.

#. Type ``reopen`` and choose **Dev Containers: Reopen in Container**:

    .. image:: ../_static/img/dev-env/13-reopen-in-container.png

#. Select a devcontainer.json file according to your platform (``.devcontainer/l4t/devcontainer.json`` for Jetson, ``.devcontainer/x86/devcontainer.json`` for x86):

    .. image:: ../_static/img/dev-env/14-select-devcontainer.png


#. Wait until the container is built and the project is opened. The remote host in the Status Bar should indicate that you are working in the container:

    .. image:: ../_static/img/dev-env/15-remote-host-status-bar.png

#. Launch the module by opening the ``run.py`` script and choosing **Terminal > Run Active File** or by clicking the ``Run and Debug`` icon in the Activity Bar.  At the end you will see pipeline's output with metadata:

    .. image:: ../_static/img/dev-env/16-run-python-file.png
