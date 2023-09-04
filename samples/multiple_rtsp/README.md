# Multiple RTSP Streams Demo

A simple pipeline demonstrates how multiplexed processing works in Savant. In the demo, two RTSP streams are ingested in the module and processed with the PeopleNet model. 

The resulting streams can be accessed via LL-HLS on `http://locahost:888/stream/city-traffic` and `http://locahost:888/stream/town-centre`.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/multiple_rtsp
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# visit 'http://127.0.0.1:888/stream/city-traffic' and 'http://127.0.0.1:888/stream/town-centre' to see how it works

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```
