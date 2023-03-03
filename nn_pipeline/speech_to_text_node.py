#from transformers import AutoTokenizer, GPTJForCausalLM
from audio_utils import pyaudio_get_input_mic_device, whisper_to_text, SpeechToTextHandler 
import whisper
import torch
import time
import rospy
from std_msgs.msg import String, Float32, Int16
from std_srvs.srv import Trigger


class Ros_S2T_talker:
    def __init__(self):
        rospy.init_node("S2T_node", anonymous = True)
        model = whisper.load_model("base")
        in_device_info = pyaudio_get_input_mic_device(searching_for ="Logitech")
        #device_index = in_device_info['index']
        device_index = 8
        #return 0
        self.s2t = SpeechToTextHandler(model = model, device = device_index,
                                       audio_input_callback = self.handle_s2t_output)

        self.publisher = rospy.Publisher("S2T_output", String, queue_size = 5)
        self.publisher_to_llm = rospy.Publisher("LLM_input", String, queue_size=1)
        
        #self.service = rospy.Service('input_llm_prompt', Trigger, self.handle_llm_prompt_ros)
        print("S2T service initialized")
        try:
            self.s2t.stream_from_mic()
            rospy.spin()
        except KeyboardInterrupt:
            self.st2.vad_audio.destroy()
            return 0
    def handle_s2t_output(self, text):
         
        out_msg = String()
        out_msg.data = text
        self.publisher.publish(out_msg)
        self.publisher_to_llm.publish(out_msg)
        return text

if __name__=="__main__":
    obj = Ros_S2T_talker()
    """
    import argparse
    parser = argparse.ArgumentParser(description="Get answer to prompt from LLM (GPT-J)")

    parser.add_argument('-p', '--prompt', type=str, default = "Some aliens were found",
                        help= "Set input prompt to the language model")
    parser.add_argument('-e', '--evil', type=bool, default = False,
                        help="GPT-J's evil twin")

    ARGS = parser.parse_args()
    #if ARGS.prompt: prompt = ARGS.prompt.value
    #else: prompt = "Some aliens were found V2"
    #if ARGS.evil: evil = ARGS.evil.value
    #else: evil = False
    evil = False
    #evil = ARGS.evil
    prompt = ARGS.prompt
    print("ARGs:")
    print("Evil:", evil)
    print("Prompt:", prompt)
    llm = LLM_wrapper(evil = evil)
    
    output = llm.eval_prompt(prompt)
    """

  
        
        
