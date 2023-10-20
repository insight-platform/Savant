VS Code
=======

In general, follow instructions from `<https://code.visualstudio.com/docs/devcontainers/containers>`__. The following steps are specific developing a Savant module starting from the module template that already contains the devcontainer configuration.

IDE Preparation
---------------

#. Install the `Remote Development <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_ extension pack.

(Optional) Connect to a remote host
-----------------------------------

#. In case you want to develop on a remote host, you can connect to it using SSH. Follow the instructions from `<https://code.visualstudio.com/docs/remote/ssh>`__ to setup SSH host and connect to it. It is recommended to setup SSH keys to avoid entering password every time you connect to the host.

Project Preparation
-------------------

#. Clone the Savant repo:

    .. code-block:: bash

        git clone https://github.com/insight-platform/Savant.git

#. Copy and rename the template (let's name the new project ``my-module``):

    .. code-block:: bash

        cp -r Savant/samples/template my-module

#. Run the IDE and open the ``my-module`` folder.

#. Update the ``devcontainer.json`` file according to the folder name and your platform (``.devcontainer/l4t/devcontainer.json`` for Jetson, ``.devcontainer/x86/devcontainer.json`` for x86):

   * Update ``--network`` value in ``runArgs``

      .. code-block:: json

        "runArgs": [ "--gpus=all", "--network=my-module_network" ],


   * Update zmq sockets volume source in ``mounts``

      .. code-block:: json

        {
            "source": "my-module_zmq_sockets",
            "target": "/tmp/zmq-sockets",
            "type": "volume"
        },

Deploy Jaeger service
---------------------

Sample module and client are configured to send traces to Jaeger service by default. Run the following command to deploy Jaeger service:

.. code-block:: bash

    docker compose -f docker-compose.x86.yml up jaeger -d

Docker Compose takes care of creating the network and volume required for communication.

Reopen in Container
-------------------

#. The following steps apply to both local and remote development (screenshots are made for remote development).

#. Open the Command Palette with **F1** or **Ctrl+Shift+P**.

#. Type ``reopen`` and select **Dev Containers: Reopen in Container**:

    .. image:: ../_static/img/dev-env/13-reopen-in-container.png

#. Select a devcontainer.json file according to your platform (``.devcontainer/l4t/devcontainer.json`` for Jetson, ``.devcontainer/x86/devcontainer.json`` for x86):

    .. image:: ../_static/img/dev-env/14-select-devcontainer.png

#. Wait until the container is built and the project is opened. The remote host in the Status Bar should indicate that you are working in the container:

    .. image:: ../_static/img/dev-env/15-remote-host-status-bar.png

#. Launch the module by opening the ``run.py`` script and choosing **Terminal > Run Active File** or by clicking the ``Run and Debug`` icon in the Activity Bar.  At the end you will see pipeline's output with metadata:

    .. image:: ../_static/img/dev-env/16-run-python-file.png

#. Open the ``client.py`` script and run it by selecting ``Run Python File in Dedicated Terminal``. You will see the client's output:

    .. image:: ../_static/img/dev-env/17-run-client.png

#. Check the results:

   * Open ``output/result_img.jpeg`` in the IDE Explorer to see the result image.

     .. image:: ../_static/img/dev-env/18-check-result-img.png

   * Visit ``http://127.0.0.1:16686`` to access the Jaeger UI.

That's it, the environment is set up. Now you are ready to develop your own pipeline. See the next section to find out how.

.. youtube:: 0u69E8sD3rE

Update Runtime On Container Change
----------------------------------

.. include:: ../includes/getting_started/1_vscode_update_docker.rst

In the following sections, you will find additional details on module development.
