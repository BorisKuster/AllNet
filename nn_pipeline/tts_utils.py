import audioop
import librosa
import numpy as np
import pyaudio
from TTS.api import TTS
import wave

class TextToSpeech:
    def __init__(self, device_info):
        # Running a multi-speaker and multi-lingual model

        # List available 
        model_name = TTS.list_models()[0]
        # Init TTS
        self.tts = TTS(model_name, progress_bar = False, gpu = True);
        #self.tts.synthesizer.output_sample_rate = int(device_info['defaultSampleRate'])
        print("TTS using device info:", device_info)
        self.device_info = device_info
        self.device_index = device_info['index']
        self.output_device_rate = 48000
    
    def play_audiofile(self, wav = None, audiofile=None):
        # length of data to read.
        chunk = 1024

        '''
        ************************************************************************
              This is the start of the "minimum needed to read a wave"
        ************************************************************************
        '''
        # open the file for reading.
        if wav is not None:
            stream = wav
        else:
            #wf, samplerate = librosa.load(audiofile, sr = 48000)
            wf = wave.open(audiofile, 'rb')
            data = wf.readframes(chunk)
            samplewidth= wf.getsampwidth()
            channels = wf.getnchannels()
            rate = wf.getframerate()
            if rate != self.output_device_rate:
                rate = self.output_device_rate
                cvstate = None
                data, cvstate = audioop.ratecv(
                    data, samplewidth, channels, rate,
                    self.output_device_rate, cvstate)
            print("Wave sample rate after reading:", rate)
             
        
        # Resample if needed
        #NOTWORKING wf = self.tts.utils.vad.resample_wav(wf

        # create an audio object
        p = pyaudio.PyAudio()

        # open stream based on the wave object which has been input.
        stream = p.open(format =
                        p.get_format_from_width(samplewidth),
                        channels = channels,
                        rate = rate,
                        output = True,
                        output_device_index = self.device_index
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

    def text_to_speech(self, text):
        from TTS.utils.vad import read_audio, resample_wav
        self.text_to_speech_to_file(text, output = 'tmp1.wav')
        
        wav,signalrate = read_audio("tmp1.wav")
        #print("SAMPLERATE", signalrate)
        #wav2 = resample_wav(wav, sr =signalrate, new_sr = self.output_device_rate )
        self.play_audiofile(audiofile = 'tmp1.wav')
        #self.play_audiofile(wav = wav2)
 
    def text_to_speech_to_file(self, text, output = 'output.wav'):
        # Run TTS
        wav = self.tts.tts(text = text)
        #librosa.output.write_wav()
        
        #self.tts.tts_to_file(text=text, speaker=self.tts.speakers[4],       language=self.tts.languages[0], 
        #                     file_path=output)
        #wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[1], language=tts.languages[0])
        # Text to speech to a file
        #tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

