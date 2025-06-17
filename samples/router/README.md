# Simple Pipeline For Demonstrating Router Functionality

The pipeline routes streams on two sinks:

- keyframes are routed to a screenshot maker module with minimum configured interval;
- all frames are routed to video archiving sink adapter.

## How to run

```bash
docker compose -f samples/router/docker-compose.yml up
```

## Results

Video archive is in `../data/output/footage/<stream_name>`, screenshots are in `../data/output/screenshots`. 