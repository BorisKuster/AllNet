from cv_bridge import CvBridge

import os 
from PIL import Image as PilImage
import requests
import torch
import io

from transformers import (
    Blip2Processor,
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)
import time
import rospy
from std_msgs.msg import String, Float32, Int16, Bool
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

class LLM_and_vision_wrapper:
    def __init__(self, evil = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Common for visual question answering, captioning and text generation from prompt
        #model_name = "Salesforce/blip2-opt-2.7b"
        #model_name = "Salesforce/blip2-opt-6.7b"
        model_name = "Salesforce/blip2-flan-t5-xxl"
        load_8bit = True
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, load_in_8bit=load_8bit,
            device_map="auto"
        )
        # .to does not work if loading 8bit version, since the model is already put on device
        if not load_8bit:self.model.to(self.device)
        
        self.temperature = 0.9
        self.min_length = 50
        self.max_length = 500     
        rospy.loginfo("Loaded model {}".format(model_name))

    def eval_prompt(self, prompt=None, image = None):
        if prompt is not None: assert type(prompt) == str
        # Three options:
        # 1. No image, only text prompt
        # 2. Image, no text prompt (image captioning)
        # 3. Image and text prompt (VQA - visual question answering)
        start_time = time.time()

        # Option 1: No image, only text prompt
        if image is None:
            assert prompt is not None
            prompt = prompt # Example: "Continue the story: The brown horse was walking along the meadow when he"
            input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            out_ids = self.model.language_model.generate(input_ids, min_new_tokens = self.min_length,
                                                         max_new_tokens = self.max_length)
            generated_text = self.processor.tokenizer.batch_decode(out_ids)[0].strip()
        
        # Option 2: Image, no text prompt (image captioning)
        elif (prompt is None) and (image is not None):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, min_new_tokens = self.min_length,
                                                max_new_tokens = self.max_length)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Option 3: Image and text prompt (VQA - visual question answering)
        elif len(prompt)>=2 and (image is not None):
            prompt = prompt # Prompt should be for example "Question: Where in the image is the battery ? Answer:"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, min_new_tokens = self.min_length,
                                               max_new_tokens = self.max_length)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            rospy.loginfo("Incorrect input format to the LLM ! ")

        time_required = time.time()-start_time
        print("LLM token generation took %.3f s"%(time_required))
        print("Prompt:", prompt)
        print("Out:", generated_text)
        
        return generated_text

class Ros_llm_vision_talker:
    def __init__(self, camera_input_topic = "/realsense/color/image_raw"):
        rospy.init_node("LLM_node", anonymous = True)
        
        self.bridge = CvBridge()          

        self.llm = LLM_and_vision_wrapper(evil=False)

        self.llm_temp_listener = rospy.Subscriber("LLM_temperature", Float32, self.change_temp_callback)
        self.llm_temp_listener = rospy.Subscriber("LLM_temperature", Float32, self.change_temp_callback)
        self.llm_max_length_listener = rospy.Subscriber("LLM_max_len", Int16, self.change_max_length_callback)
        self.llm_min_length_listener = rospy.Subscriber("LLM_min_len", Int16, self.change_min_length_callback)
        self.llm_use_image_listener = rospy.Subscriber("LLM_use_image", Bool, self.change_use_image_callback)

        self.publisher = rospy.Publisher("LLM_output", String, queue_size = 5)
        self.listener = rospy.Subscriber("LLM_input", String, self.handle_llm_prompt_ros)

        self.camera_listener = rospy.Subscriber(camera_input_topic, Image, self.handle_image)
        
        self.use_image = True
        self.save_image = True # Save image for debugging purposes
        self.saved_image_name = 'LLM_input.jpg'
        #self.service = rospy.Service('input_llm_prompt', Trigger, self.handle_llm_prompt_ros)
        print("LLM and vision service initialized")
        rospy.spin()

    def handle_image(self, input):
        if self.use_image:
            self.last_image = input 
    def change_use_image_callback(self, input):
        self.use_image = input.data
    def change_min_length_callback(self, input):
        if (input.data>0) and type(input.data)==int:
            self.llm.min_length = input.data
    def change_max_length_callback(self, input):
        if (input.data>0) and type(input.data)==int:
            self.llm.max_length = input.data 
    def change_temp_callback(self, input):
        if (input.data > 0) and (input.data<=1):
            self.llm.temperature = input.data

    def handle_llm_prompt_ros(self, prompt):
        # Three cases:
        # 1. No image, only text prompt
        # 2. Image, no text prompt (image captioning)
        # 3. Image and text prompt (VQA - visual question answering)
        prompt = prompt.data
        
        # Case 1
        if self.use_image == False:
            generated_text = self.llm.eval_prompt(prompt = prompt, image = None)
        # Case 2
        elif self.use_image and len(prompt)<2:
            image = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding='rgb8')
            pil_image = PilImage.fromarray(image)
            pil_image.save("seen_img_2.jpg")
            #redpilled_image =  PilImage.open(io.BytesIO(bytearray(self.last_image)))
            generated_text = self.llm.eval_prompt(prompt = None, image = image)
        # Case 3
        elif self.use_image and len(prompt)>=2:
            prompt = "Question: {} ? Answer:".format(prompt)
            image = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding='rgb8')
            pil_image = PilImage.fromarray(image)
            pil_image.save("seen_img.jpg")
            #redpilled_image =  PilImage.open(io.BytesIO(bytearray(self.last_image)))
            generated_text = self.llm.eval_prompt(prompt = prompt, image = image)
            
        out_msg = String()
        out_msg.data = generated_text
        self.publisher.publish(out_msg)
        return generated_text

if __name__=="__main__":
    obj = Ros_llm_vision_talker()
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

  
        
        
