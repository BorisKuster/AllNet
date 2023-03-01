import whisper
import ffmpeg
import numpy as np
import torch
import pyaudio
import wave
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import webrtcvad
from halo import Halo
from scipy import signal
import sys
logging.basicConfig(level=20)

def play_audiofile(audiofile=None, output_device_index = 0):
    # length of data to read.
    chunk = 1024

    '''
    ************************************************************************
          This is the start of the "minimum needed to read a wave"
    ************************************************************************
    '''
    # open the file for reading.
    wf = wave.open(audiofile, 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True,
                    output_device_index = output_device_index
                    )

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)


    # cleanup stuff.
    wf.close()
    stream.close()    
    p.terminate()

def pyaudio_get_input_mic_device(searching_for = "Logitech"):
    # Input params
    # Searching for - partial name of device which should be used. UPPER AND LOWER CASE MATTERS ! 

    # Find device
    good_device_info = None
    aud = pyaudio.PyAudio()
    for i in range(0,15):
        try:
            info = aud.get_device_info_by_index(i)
            #print(info['name'])
            if searching_for in info['name']:
                print(info)
                good_device_info = info
        except Exception as e :
            print("Exception:", e)
            
    aud.terminate()
    if good_device_info is None:
        raise Exception("Did not find a good input device !")
    else:
        return good_device_info

def record_clip(DEVICE_INDEX = 10, FORMAT = pyaudio.paInt16, RATE=48000, CHUNK=1024, RECORD_SECONDS=6,
                WAVE_OUTPUT_FILENAME = 'filename.wav'):

    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=DEVICE_INDEX)
                        # sample_rate=RATE)
    print("recording...")
    print('---------------------------------')
    print(int(RATE / CHUNK * RECORD_SECONDS))
    print('*********************************')

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        print("Recording . . .")

        frames.append(data)
    print("Recording finished. . .")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(2)
    audio.get_sample_size(FORMAT)
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    
def whisper_to_text(whisper_model = None, audiofile = None, steamed_frame = None, language = 'english'):
    if audiofile is not None:
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audiofile)
    else:
        audio = steamed_frame
   
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False, language = language)
    result = whisper.decode(whisper_model, mel, options)

    # print the recognized text
    print(result.text)
    output = result.text
    return output

#whisper_to_text(audiofile =None, whisper_model = whisper_model, steamed_frame = out)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')
        print("KWARGS:", kwargs)
        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech
        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tobytes()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

class SpeechToTextHandler:
    def __init__(self, nospinner = True, model=None, savewav = False, vad_aggressiveness=3, device = 7,
                 rate = 48000,file = 'out.wav'):
        
        if model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(self.device)
            self.model = whisper.load_model("base").to(self.device)
        
        # Start audio with VAD
        self.vad_audio = VADAudio(aggressiveness=vad_aggressiveness,
                             device=device,
                             input_rate=rate,
                             file=file)
        self.model = model

        # Stream from microphone to the model using VAD
        self.spinner = None
        if not nospinner:
            self.spinner = Halo(spinner='line')
            
        self.all_frames = []
        
        self.savewav = savewav
        
    def stream_from_mic(self):
        print("Listening (ctrl-C to exit)...")
        self.frames = self.vad_audio.vad_collector()
        try:
            wav_data = bytearray()
            for frame in self.frames:
                if frame is not None:
                    if self.spinner: self.spinner.start()
                    logging.debug("streaming frame")
                    cur_frame = np.frombuffer(frame, np.int16).flatten().astype(np.float32) / 32768.0
                    self.all_frames.append(cur_frame)
                    #txt = whisper_to_text(whisper_model = model, audiofile = None, steamed_frame = cur_frame)
                    #print(txt)
                    #stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
                    if self.savewav: wav_data.extend(frame)
                else:
                    if self.spinner: self.spinner.stop()
                    logging.debug("end utterance")
                    if self.savewav:
                        self.vad_audio.write_wav(os.path.join(self.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                        wav_data = bytearray()
                    #text = stream_context.finishStream()
                    try:
                        self.all_frames = np.array(self.all_frames)
                        #all_frames = all_frames[0:10,:]
                        self.all_frames = np.hstack(self.all_frames)
                        #print(self.all_frames.shape)
                        text = whisper_to_text(whisper_model = self.model, audiofile = None,
                                               steamed_frame = self.all_frames)
                        #print("Recognized: %s" % text)
                        #stream_context = model.createStream()
                        #print("END utterance, ", type(all_frames[0]))
                    except Exception as e:
                        print("Exception ", e)
                        self.vad_audio.destroy()
                    self.all_frames = []
        except KeyboardInterrupt:
            self.vad_audio.destroy()
            return self.vad_audio
