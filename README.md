# AllNet
A combined pipeline to handle inputs such as speech, text and images.

# Some sample commands:

arecord -d 2 -r 48000 foobar.wav --device=hw:3,0 --format=S16_LE

aplay foobar.wav --device=plughw:CARD=Headset,DEV=0

python3 mic_vad_streaming.py --model='/deepspeech-0.9.3-models.pbmm' --rate=44100
