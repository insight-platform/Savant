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

Reopen in container
-------------------

#. The following steps apply to both local and remote development (screenshots are made for remote development).

#. Open the command palette with **F1** or **Ctrl+Shift+P**.

#. Type ``reopen`` and choose **Dev Containers: Reopen in Container**:

    .. image:: ../_static/img/dev-env/13-reopen-in-container.png

#. Select a devcontainer.json file according to your platform (``.devcontainer/l4t/devcontainer.json`` for Jetson, ``.devcontainer/x86/devcontainer.json`` for x86):

    .. image:: ../_static/img/dev-env/14-select-devcontainer.png


#. Wait until the container is built and the project is opened. The remote host in the Status Bar should indicate that you are working in the container:

    .. image:: ../_static/img/dev-env/15-remote-host-status-bar.png

#. Launch the module by opening the ``run.py`` script and choosing **Terminal > Run Active File** or by clicking the ``Run and Debug`` icon in the Activity Bar.  At the end you will see pipeline's output with metadata:

    .. image:: ../_static/img/dev-env/16-run-python-file.png
