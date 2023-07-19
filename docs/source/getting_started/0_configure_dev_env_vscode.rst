VS Code
=======

IDE Preparation
---------------

#. Install the `Remote Development <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_ extension pack.

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

#. Open the command palette with **F1** or **Ctrl+Shift+P**.

#. Type ``reopen`` and choose **Dev Containers: Reopen in Container**:

    .. image:: ../_static/img/dev-env/13-reopen-in-container.png

#. Select a devcontainer.json file according to your platform (``.devcontainer/l4t/devcontainer.json`` for Jetson, ``.devcontainer/x86/devcontainer.json`` for x86):

    .. image:: ../_static/img/dev-env/14-select-devcontainer.png


#. Wait until the container is built and the project is opened. The remote host in the Status Bar should indicate that you are working in the container:

    .. image:: ../_static/img/dev-env/15-remote-host-status-bar.png

#. Launch the module by opening the ``run.py`` script and choosing **Terminal > Run Active File** or by clicking the ``Run and Debug`` icon in the Activity Bar.  At the end you will see pipeline's output with metadata:

    .. image:: ../_static/img/dev-env/16-run-python-file.png
