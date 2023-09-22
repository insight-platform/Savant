
Changes to the Dockerfile or the base image
```````````````````````````````````````````

You should check that the "Rebuild image automatically every time before running code" option is enabled in the "Python interpreters" settings. If this option is enabled and you have made any changes to the Dockerfile or updated the base image, you don't need to do anything else. At the next run a new image will be built and the container will be updated.

.. image:: ../_static/img/dev-env/03-setup-docker-build.png
    :width: 500


Adding new packages to requirements.txt
```````````````````````````````````````

Once dependencies are added to the ``requirements.txt``, they will be automatically installed when building a new image.

However, PyCharm does not automatically detect newly installed packages in the Docker container. The PyCharm will update the skeleton at the next startup or you can manually scan for new packages. To do this, you need to open the **Settings** and look for **Rescan**, then navigate to **Plugins > Python > Rescan Available Python Modules and Packages** and set the hotkey (e.g., **Alt+R**):

.. image:: ../_static/img/dev-env/12-rescan.png
    :width: 500

After adding a new package to the ``requirements.txt``, simply press the specified hotkey to rebuild the image and update the packages.
