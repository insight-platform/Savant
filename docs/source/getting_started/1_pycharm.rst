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

#. When creating a project, select the system Python interpreter

.. image:: ../_static/img/dev-env/01-creating-project.png

Setting Up The Interpreter
--------------------------
#. Open the project settings with **File > Settings ...** or **Ctrl+Alt+S**. Set **Use compose V2** in **Build, Execution,Deployment > Docker > Tools**

    .. image:: ../_static/img/dev-env/21-select-docker-compose-v2.png

#. Open the project settings with **File > Settings ...** or **Ctrl+Alt+S**.

#. Choose **Project: my-module > Python Interpreter** section.

#. Add a new **Python Interpreter** by clicking the **Add Interpreter**, and choosing **On Docker Compose...**:

    .. image:: ../_static/img/dev-env/02-setup-python-interpreter.png

#. In the modal window:

  * **[step 1 of 4]**: Click on the three dots under **Server** and set up access to the docker service

    .. image:: ../_static/img/dev-env/20-setup-docker.png

  * **[Step 2 of 4]**: Choose Configuration file: ``docker-compose.x86.yml`` or ``docker-compose.l4t.yml``, if you are on Jetson, and select the service **module**:

    .. image:: ../_static/img/dev-env/03-setup-python-interpreter-docker-compose-module.png

  * **[Step 3 of 4]**: Click **Next** button, make sure docker build is successful and then again click **Next** button.

  * **[Step 4 of 4]**: Keep the default python interpreter (python3) and click **Create**:

    .. image:: ../_static/img/dev-env/04-setup-interpreter-in-docker.png

#. Make sure that the ``savant`` package is in the package list, click **OK**:

    .. image:: ../_static/img/dev-env/05-check-savant-package.png

#. Add a another **Python Interpreter** by clicking the **Add Interpreter**, and choosing **On Docker Compose...**:

    .. image:: ../_static/img/dev-env/02-setup-python-interpreter.png

#. In the modal window:

  * **[Step 1 of 3]**: Choose Configuration file: ``docker-compose.x86.yml`` or ``docker-compose.l4t.yml``, if you are on Jetson, and select the service **client**:

    .. image:: ../_static/img/dev-env/03-setup-python-interpreter-docker-compose-module.png

  * **[Step 2 of 3]**: Make sure docker build is successful and then click **Next** button.

  * **[Step 3 of 3]**: Keep the default python interpreter (python3) and click **Create**:

    .. image:: ../_static/img/dev-env/22-setup-python-interpreter-docker-compose-client.png

#. Open the ``my-module/run.py`` file in the IDE.

#. Wait for IDE to update the skeletons and check that IDE `sees` dependencies pointing the mouse over the line with the ``import`` directive: the popup must appear with the description of the function:

    .. image:: ../_static/img/dev-env/06-check-func-spec.png

Configure Module Run
--------------------

The ``module/run.py`` file is the entrypoint of the module, let's configure the launch command for the script.

#. Open the ``module/run.py`` script

#. Right click of mouse on the tab ``module/run.py`` and choose **Modify Run Configuration** in ""More Run/Debug:

    .. image:: ../_static/img/dev-env/23-modify-run-configureation.png

#. In the **Environment** section add ``PYTHONPATH=/opt/savant`` (PyCharm `rewrites <https://youtrack.jetbrains.com/issue/PY-32618/The-original-PYTHONPATH-is-replaced-by-PyCharm-when-running-configurations-using-Docker-interpreter>`_  ``PYTHONPATH``), select **Docker compose(module)** interpreter. If you used a custom Dockerfile, you need to set ``up --build module`` in **Docker Compose > Command and options**.

    .. image:: ../_static/img/dev-env/08-run-config-module.png

#. Launch the module by choosing **Run 'run'** or with the hotkey **Ctrl+Shift+F10**:

    .. image:: ../_static/img/dev-env/10-run-output-1.png

#. You may see various GStreamer error messages: it's ok. At the end you will see the output ``module status to ModuleStatus.RUNNING.``:

    .. image:: ../_static/img/dev-env/11-run-output-2.png

Configure Client Run
--------------------

The ``client/run.py`` This is a script that allows you to interact with the module by sending data and receiving results., let's configure the launch command for the script.

#. Open the ``client/run.py`` script

#. Right click of mouse on the tab ``client/run.py`` and choose **Modify Run Configuration** in ""More Run/Debug:

    .. image:: ../_static/img/dev-env/23-modify-run-configureation.png

#. In the **Environment** section add ``PYTHONPATH=/opt/savant`` (PyCharm `rewrites <https://youtrack.jetbrains.com/issue/PY-32618/The-original-PYTHONPATH-is-replaced-by-PyCharm-when-running-configurations-using-Docker-interpreter>`_  ``PYTHONPATH``) and select **Docker compose(client)** interpreter:

    .. image:: ../_static/img/dev-env/09-run-config-client.png

#. Launch the client script by choosing **Run 'run client'** or with the hotkey **Ctrl+Shift+F10**:

    .. image:: ../_static/img/dev-env/25-run-client.png

#. Check the results:

   * Open ``output/result_img.jpeg`` in the IDE Explorer to see the result image.

     .. image:: ../_static/img/dev-env/24-check-result-img-pycharm.png

   * Visit ``http://127.0.0.1:16686`` to access the Jaeger UI.

That's it, the environment is set up. Now you are ready to develop your own pipeline. See the next section to find out how.

Update Runtime On Container Change
----------------------------------

.. include:: ../includes/getting_started/1_pycharm_update_docker.rst

In the following sections, you will find additional details on module development.
