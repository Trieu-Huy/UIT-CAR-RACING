
# UIT-CAR-RACING
# Unity Autonomous Road Segmentation & Navigation


### Overview
This project demonstrates an AI-powered car that performs **road segmentation and automatic path following** in a **Unity 3D simulation environment**.

Developed as part of an [AI-powered automatic path-following competition](https://www.facebook.com/share/p/17i8UeiL9j/) in my university, it integrates computer vision and control algorithms to simulate autonomous driving behavior. (WARNING: This project is mostly a product of vibe coding)

#####  Setup

This project consists of three main components:
1. **Unity Simulation**        –   environment and vehicle physics
2. **CEEC Docker Container**   –   communication bridge between Unity and Python
3. **Python Controller**       –   YOLOv8-based segmentation and automatic steering

Follow the steps below to set everything up.

---

### Prerequisites

Make sure you have the following installed:

- **Windows 10/11** or **Ubuntu 20.04+** (perferably)
- **Python 3.8 – 3.10**
- **Visual Studio Code** (Versions before March 2025)
- **Docker environment** [(web)](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04)
- **Unity Map** [(drive)](https://drive.google.com/file/d/1On6iAmioqvXPbQl20_R3ndLwDB5msQj9/view)
```bash
docker pull quocle28/it_car_2023:v1
```

For GPU acceleration (optional):
- **NVIDIA Container Toolkit** [(web)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **NVIDIA drivers** (Ubuntu)

Visual Studio Code extensions:
- **Docker**
- **Dev container**
- **Pip installations**:
```bash
  pip install ultralytics opencv-python numpy torch torchvision tqdm matplotlib
```

### Start
#### 1. Pull the docker:
```bash
docker pull quocle28/it_car_2023:v1
```
#### 2. Attach the docker to VScode:

Run this line on bash / powershell to attach the docker environment to VScode.
```bash
docker run --name it-car -it -p 11000:11000 --network="host" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all <imageid>
```
Change "imageid" to the image id.

#### 3. Attach the model to the container:

Run this line on bash / powershell.
```bash
docker cp <model_path> <container_id>:/workspace/
```
Change "model_path" and "container_id to the model path and container id.

#### 4. Run:
Turn on "Run as executable" in the Unity file properties.
Run the following command in VScode to run the project:
```
  python maycay.py
```
<<<<<<< HEAD
=======
# UIT-CAR-RACING
>>>>>>> 683791e (Initial commit)
=======
>>>>>>> 152320a81da450ae7fab50677db8e5cddf9f4a6f
