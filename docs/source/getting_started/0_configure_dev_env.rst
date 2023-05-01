Configure The Development Environment
=====================================

Savant requires a complex environment (GStreamer, DeepStream, OpenCV). Installing and configuring all the necessary packages is not a trivial task. We suggest using prepared docker images for working with Savant (both for development and for production). You get an isolated and fully configured environment in a few clicks.

Popular Python IDEs, PyCharm and VSCode support the ability to set up a development environment with docker.

The Savant repository has a module template for a quick start, we will use it and show you how to set up a convenient development environment in popular IDEs.

PyCharm Professional
--------------------

You can use the official documentation `Configure a docker interpreter <https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html>`_, but there are nuances. Below is our How-to screenlist with explanations.

The instruction is relevant for PyCharm 2023.1 (Professional Edition).

Project preparation
^^^^^^^^^^^^^^^^^^^

#. Clone Savant repo, copy and rename model template (let's name the new project `my-module`)

    .. code-block:: bash

        cp -r Savant/samples/template my-module

#. Run the IDE and open your new project

#. When opening the project, skip the stage of creating the environment - click `Cancel` on `Creating Virtual Environment`

.. image:: ../_static/img/dev-env/01-cancel-creating-virtual-environment.png

Setting up an interpreter
^^^^^^^^^^^^^^^^^^^^^^^^^

#. Open the project settings via the Menu `File > Settings...` or `Ctrl+Alt+S`

#. Choose `Project: my-module > Python Interpreter` section

#. Add new `Python Interpreter` by clicking the `Add Interpreter`, and choosing `On Docker...`

    .. image:: ../_static/img/dev-env/02-setup-python-interpreter.png

#. In the opened modal window, on the Step 1/3: Choose Dockerfile (docker/Dockerfile.x86 or l4t, if you are on Jetson) and set up Context folder (should be the root of the project). In the Optional section you can enter an `Image tag` (it will appear in interpreter name)

    .. image:: ../_static/img/dev-env/03-setup-docker-build.png

#. Step 2/3: Make sure docker build is successful

#. Step 3/3 Leave the default python interpreter (python3), click `Create`

    .. image:: ../_static/img/dev-env/04-setup-interpreter-in-docker.png

#. Make sure `savant` is in the package list, click OK

    .. image:: ../_static/img/dev-env/05-check-savant-package.png

#. Open the `module/run.py` file in the IDE

#. Check that when you hover over the line with `import`, a popup appears with a description of the function

    .. image:: ../_static/img/dev-env/06-check-func-spec.png

Configure module run
^^^^^^^^^^^^^^^^^^^^

The `run.py` is the entrypoint of the module, let's prepare the launch of the script in the container.

#. Click on the `run` icon and choose `Modify Run Configuration`

    .. image:: ../_static/img/dev-env/07-open-run-config.png

#. In the `Environment` section add `PYTHONPATH=/opt/savant` (`PyCharm rewrites PYTHONPATH <https://youtrack.jetbrains.com/issue/PY-32618/The-original-PYTHONPATH-is-replaced-by-PyCharm-when-running-configurations-using-Docker-interpreter>`_)

    .. image:: ../_static/img/dev-env/08-run-config-env.png

#. In the `Docker container settings` section add `gpus all`

    .. image:: ../_static/img/dev-env/09-run-config-ops.png

#. Launch the module by choosing `Run 'run'` or `Ctrl+Shift+F10`

    .. image:: ../_static/img/dev-env/10-run-output-1.png

#. There will be some GStreamer messages, don't worry, it's ok. At the end you will see pipeline console output with metadata

    .. image:: ../_static/img/dev-env/11-run-output-2.png

That's it, the environment is set up. Now you are ready to start implementing your own pipeline: modify the module config (`module/module.yaml`), add your own components, etc.

Notes
^^^^^
PyCharm does not automatically collect information about newly installed packages in the Docker container. However, there is an option to manually scan for new packages. To do this, go to the `Settings` and search for "rescan", then navigate to `Plugins -> Python -> Rescan Available Python Modules and Packages` and set the hotkey. For example `Alt+R`

.. image:: ../_static/img/dev-env/12-rescan.png

After adding a new package to the `requirements.txt`, simply press the specified hotkey to rebuild the Docker image and update the packages.
