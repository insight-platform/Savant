# Savant: Performing People Detection, Face Detection (and Blurring) and Tracking

Savant is framework build over Nvidia's DeepStream that provides a flexible solution to provide real time video analytics solution on GPU and edge devices, at much lower latency than using frameworks like PyTorch and Tensorflow.

This particular fork of Savant implements peeople detection and tracking using input video file.

# Specifications
Platform: Linux (Ubuntu 22)

Detector: [PeopleNet (Nvidia)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)


Tracker: [NvDeepSORT (Nvidia)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)



# Instructions

## 1. Environment Setup

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl -y

# install docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# intall nvidia driver
sudo apt install --no-install-recommends nvidia-driver-530
sudo reboot

# install nvidia container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker


# (Optional) to test if above installation is working correctly
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

```


## 2. Place the target video (e.g: ref_vid.mp4) to suitable directory

``` bash
git clone https://github.com/antidianuj/Savant.git
cd Savant/samples/peoplenet_detector
git lfs pull
sudo mv /path/to/ref_vid.mp4 data
```
If the target video file is of another name, then change the above code accordingly, as well as the name of file in 'docker-compose.x86.yml' in Savant/samples/peoplenet_detector.




## 3. Run the detector and tracker:

```bash
sudo docker compose -f docker-compose.x86.yml up

# view the output result by visiting following link in browser
# http://127.0.0.1:888/stream/

```


# Example Result

https://github.com/antidianuj/Savant/assets/47445756/3dff2000-343a-45dd-855f-3db2be1b9947



