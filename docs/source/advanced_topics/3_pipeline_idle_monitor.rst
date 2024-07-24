Pipeline Idle Monitor
======================

In some cases, it may be necessary to actively monitor the pipeline for periods of inactivity.
The Pipeline Idle Monitor is a monitoring tool designed to observe the activity and performance of data pipelines, ensuring they remain operational and effective.


Design
-------

The Pipeline Idle Monitor is a separate component within the pipeline topology. It uses docker client to interact with the pipeline containers.

.. warning::
	Requires access to the Docker daemon socket to manage the services in the pipeline.

Monitoring is based on the use of a `buffer adapter <https://docs.savant-ai.io/v0.4.0/savant_101/10_adapters.html#buffer-bridge-adapter>`__ that provides information on the number of incoming and outgoing messages, called metrics.
Metrics characterize services that are connected to the input and output of the buffer adapter.

The Pipeline Idle Monitor is designed to detect the following conditions based on specific metrics:

- No data has been received for a certain period, indicating that the downstream adapters are not receiving any data.
- No data has been sent for a certain period, indicating that the preceding adapters are not transmitting data.
- The buffer is overloaded, indicating that the downstream adapters cannot process the data at the expected rate.

There can be multiple buffer adapters in the pipeline, and each of them is monitored separately by a single Pipeline Idle Monitor.

.. note::
	Buffer adapter should be properly configured to use **prometheus** as metrics provider.

The source code and further details can be found in the `PipelineWatchdog <https://github.com/insight-platform/PipelineWatchdog>`__ repository.

Using the Pipeline Idle Monitor
--------------------------------

The Pipeline Idle Monitor is configured using the YAML configuration file.

The configuration file contains three main sections for a buffer adapter:

- **queue** - defines the condition for the buffer adapter to be considered overloaded
- **egress** - defines the time period for which the downstream adapters should be considered idle if no data has been sent
- **ingress** - defines the time period for which the preceding adapters should be considered idle if no data has been received

.. note::
	At least one of the **queue**, **egress**, or **ingress** sections must be defined.

Each section also defines the action to be taken when the corresponding condition is met. There are two possible actions: **restart** and **stop**.

Action is applied to the services that match the container configuration.
**Labels** field can contain a single label key, a label key with a value, or a list of one or more such values. If the list is specified, the action is applied to the services that have all the labels in the list.
If multiple **labels** fields are specified, the action is applied to the services that match at least one of the fields.

Let's consider the following example:

.. code-block:: yaml

	container:
		- labels: [label=1, label2]
		- labels: label3

In this case, the action is applied to the services that have the labels **label=1** and **label2** or the label **label3**.

.. note::
	If no containers are matched by labels, the action is considered successful, and monitoring continues.

When the action is applied, the Pipeline Idle Monitor waits for the specified period (**cooldown**) before rechecking the condition.
In the other case, it will use the **polling_interval** parameter for waiting to prevent checking metrics repeatedly in a short period.

.. tip::
	It is recommended to set the **cooldown** parameter depending on the initialisation time of services that could potentially be affected by the action, so that the service has time to restart and start processing data.

Time parameters are specified as a string with a number and a unit. The following units are supported:

- **s** - seconds
- **m** - minutes
- **h** - hours
- **d** - days
- **w** - weeks

You can specify the described parameters for one or more buffer adapters in the pipeline.

Example
^^^^^^^^

For example, you may want to automatically restart a part of the pipeline when no data has been received for a certain period of time or stop a specific service when the buffer is overloaded then the configuration file will look like this:

.. code-block:: yaml

	watch:
		- buffer: buffer1:8000
		  ingress:
			action: restart
			cooldown: 60s
			idle: 100s
			container:
				- labels: label2
				- labels: label3
		  queue:
			action: stop
			length: 999
			cooldown: 60s
			polling_interval: 10s
			container:
				- labels: label1

The sample that demonstrates typical use case and configuration is available in the `samples/pipeline_monitoring <https://github.com/insight-platform/PipelineWatchdog/tree/main/samples/pipeline_monitoring>`__ directory.


