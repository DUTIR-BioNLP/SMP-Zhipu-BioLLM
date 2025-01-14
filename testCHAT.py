import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_stream_vllm, LoRARequest, inference_vllm,VllmGenerationConfig
)
import json
from tqdm import tqdm


if __name__ == '__main__':

    # folder_path = '/data/qjw/ms-swift-main/test_data/NER'
    # file_list = os.listdir(folder_path)

    # lora_checkpoint = '/data/qjw/ms-swift-main/output/glm4-9b-chat/v6-20240924-223358/checkpoint-22400'
    # lora_request = LoRARequest('default-lora', 1, lora_checkpoint)

    model_type = ModelType.glm4_9b_chat
    llm_engine = get_vllm_engine(model_type, torch.bfloat16, enable_lora=True,
                                max_loras=1, max_lora_rank=8, tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=8192)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)
    # 与`transformers.GenerationConfig`类似的接口
    llm_engine.generation_config.max_new_tokens = 128

    # for text in texts:
    while True:
        print('question: ')
        text = input().strip()
        request_list = [{'query': text, 'history':history}]
        print("history:\n", history)
        answer = inference_vllm(llm_engine, template, request_list, 
                                    generation_config=VllmGenerationConfig(repetition_penalty=1.05,
                                                                        presence_penalty=True,
                                                                        max_tokens=500,
                                                                        temperature=0.3,
                                                                        top_p=0.7,
                                                                        top_k=20,
                                                                        skip_special_tokens=True,
                                                                        stop_token_ids=[151329,151336,151338]))[0]
        
        response = answer['response']
        history = answer['history']
        print("answer: \n", response,'\n-----------------------------------\n')
