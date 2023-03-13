# AllNet

Old repo. New code available at:
https://repo.ijs.si/bkuster/allnet



A pipeline composed of several neural network modules to handle inputs such as audio(speech), text and images/videos, and with capability to output vectors, text, or images (and videos?).

# Pre-installation config
In the dockerfile "dockerfiles/Dockerfile_allnet", change the line "nvidia_driver_major_version" to the major version (first 3 numbers) of your nvidia driver (470, 525 etc.).
By default, the package expects to reside in "/home/$USER/allnet". If not there, you must also change the volume location in "docker-compose.yml"

# Installation
Install docker engine as per 
https://docs.docker.com/desktop/install/linux-install/

Install docker compose (for ubuntu "sudo apt install docker-compose")

Install nvidia-runtime-container as per
https://github.com/NVIDIA/nvidia-container-runtime
Also do what is described in the Docker Engine setup section !

Run the build commands shown in "dockerfiles/build_commands"

# To run
In bash, run "./run_or_restart_system.sh"

# Some sample commands:

# Visual + text - openCLIP

# Audio - DeepSpeech TODO : Combine with Language Model

arecord -d 2 -r 48000 foobar.wav --device=hw:3,0 --format=S16_LE # Mic record 2seconds

aplay foobar.wav --device=plughw:CARD=Headset,DEV=0 # Test speakers and mic (playback recording)

python3 mic_vad_streaming.py --model='/deepspeech-0.9.3-models.pbmm' --rate=48000 --device=7
