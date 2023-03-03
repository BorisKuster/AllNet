from transformers import AutoTokenizer, GPTJForCausalLM
import torch
import time
import rospy
from std_msgs.msg import String, Float32, Int16
from std_srvs.srv import Trigger

class LLM_wrapper:
    def __init__(self, evil = False):

 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if evil == False:
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                         torch_dtype=torch.float16).to(self.device)
        else :
            self.model = GPTJForCausalLM.from_pretrained("gpt4chan_model_float16/pytorch_model.bin", local_files_only=True,
                                                         revision="float16", torch_dtype=torch.float16).to(self.device)
        
 
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

        self.temperature = 0.9
        self.max_length = 100        
    def eval_prompt(self, prompt):
        start_time = time.time()

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        gen_tokens = self.model.generate(input_ids,
                                do_sample=True,
                                temperature=self.temperature,
                                max_length=self.max_length,)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        time_required = time.time()-start_time
        print("LLM token generation took %.3f s"%(time_required))
        print("Prompt:", prompt)
        print("Out:", gen_text)
        
        return gen_text

class Ros_llm_talker:
    def __init__(self):
        rospy.init_node("LLM_node", anonymous = True)
        
        self.llm = LLM_wrapper(evil=False)
        
        self.llm_temp_listener = rospy.Subscriber("LLM_temperature", Float32, self.change_temp_callback)
        self.llm_max_length_listener = rospy.Subscriber("LLM_max_len", Int16, self.change_max_length_callback)

        self.publisher = rospy.Publisher("LLM_output", String, queue_size = 5)
        self.listener = rospy.Subscriber("LLM_input", String, self.handle_llm_prompt_ros)
        
        #self.service = rospy.Service('input_llm_prompt', Trigger, self.handle_llm_prompt_ros)
        print("LLM service initialized")
        rospy.spin()
    def change_max_length_callback(self, input):
        if (input.data>0) and type(input.data)==int:
            self.llm.max_length = input.data 
    def change_temp_callback(self, input):
        if (input.data > 0) and (input.data<=1):
            self.llm.temperature = input.data
    def handle_llm_prompt_ros(self, prompt):
        output = self.llm.eval_prompt(prompt.data)

        out_msg = String()
        out_msg.data = output
        self.publisher.publish(out_msg)
        return output

   
 
 


if __name__=="__main__":
    obj = Ros_llm_talker()
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

  
        
        
