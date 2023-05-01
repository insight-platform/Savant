Communication With External Systems
===================================

We consider three kinds of pipeline circuits:

- end-to-end capacity (non-realtime) circuit;
- end-to-end realtime circuit;
- mixed circuit;

Different circuits require different approaches when implementing the pipelines. We will share with you the best practices for implementing such kinds of circuits.

Pure Capacity Circuits
----------------------

Such circuits are the easiest thing to create. Purely capacity circuits assume that if a part of the circuit is stuck or has become slow, the whole pipeline can be delayed or slowed down without problems.

A typical example of such a circuit: File -> Module -> MongoDB

If Mongo is dead or slow, the processing will be delayed or stuck until it recovers. You don't have to use sophisticated logic to craft such kinds of circuits.

Our recommendations when building such sort of circuits:

- use dealer/router or req/rep sockets;
- use persistent queues like Kafka to optimize the hardware utilization.

Pure Real-Time Circuits
-----------------------

As it comes from its name, a real-time circuit works in a real-time with no delays expected. Such circuits may tolerate short-term traffic bursts with internal buffers, but suffer from the situations when a part of the circuit stuck.

An example of such a circuit: RTSP -> Module -> RTSP

So, to realize such circuits, we give following recommendations:

- use pub/sub sockets, as they drop the packets which cannot be delivered;
- use low-latency systems with predictable functionality to implement 3rd-party communications (Redis/KeyDB, ZeroMQ Pub/Sub);
- prefer UDP over TCP when possible (e.g., when exporting metrics to Graphite, choose UDP) because UDP works without connection establishing;
- don't use systems that recover for a long time;
- configure hard timeouts for all operations with 3rd-party systems;
- if non-real-time systems are involved, use queues (e.g., Redis's List, RabbitMQ, MQTT, NATS) to decouple the communication with them.

Mixed Circuit
-------------

Mixed circuits derive both properties and limitations of capacity circuits and real-time circuits. You must craft them carefully, combining `pub/sub` and `dealer/router (req/rep)` sockets where necessary, adding queues and real-time systems to overcome bottlenecks. This is a hard topic requiring skills and practice. There are no rules of thumb; you must experiment and consider various failure scenarios to design predictably performing circuits.

Such circuits definitely require skills in developing real-time circuits.

