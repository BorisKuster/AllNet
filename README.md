# AllNet
A pipeline composed of several neural network modules to handle inputs such as audio(speech), text and images/videos, and with capability to output vectors, text, or images (and videos?).

# Some sample commands:

# Visual + text - openCLIP

# Audio - DeepSpeech TODO : Combine with Language Model

arecord -d 2 -r 48000 foobar.wav --device=hw:3,0 --format=S16_LE # Mic record 2seconds

aplay foobar.wav --device=plughw:CARD=Headset,DEV=0 # Test speakers and mic (playback recording)

python3 mic_vad_streaming.py --model='/deepspeech-0.9.3-models.pbmm' --rate=48000 --device=7
