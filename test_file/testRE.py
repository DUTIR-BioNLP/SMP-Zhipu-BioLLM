import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_stream_vllm, LoRARequest, inference_vllm,VllmGenerationConfig
)
import json
from tqdm import tqdm


if __name__ == '__main__':

    folder_path = '/data/qjw/ms-swift-main/test_data/RE'
    file_list = os.listdir(folder_path)

    # lora_checkpoint = '/data/qjw/ms-swift-main/output/glm4-9b-chat/v6-20240924-223358/checkpoint-22400'
    # lora_request = LoRARequest('default-lora', 1, lora_checkpoint)

    model_type = ModelType.glm4_9b
    llm_engine = get_vllm_engine(model_type, torch.bfloat16, enable_lora=True,
                                max_loras=1, max_lora_rank=8, tensor_parallel_size=8,gpu_memory_utilization=0.9,max_model_len=8192)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)
    # 与`transformers.GenerationConfig`类似的接口
    llm_engine.generation_config.max_new_tokens = 128

    for file in file_list:
        # print(file[:-6])
        # if file != 'RE-cmeie_v2-dev.jsonl':
        #     continue
        data = []
        with open('/data/qjw/ms-swift-main/test_data/RE/'+file, 'r', encoding='utf-8') as fin:
            for line in fin:
                json_data = json.loads(line)
                data.append(json_data)
        
        new_datas = []
        for per_data in tqdm(data):
            new_data = {}
            golden_label = per_data['conversation'][0]['assistant']
            text = per_data['conversation'][0]['human']
            request_list = [{'query': text}]
            response = inference_vllm(llm_engine, template, request_list, 
                                      generation_config=VllmGenerationConfig(repetition_penalty=1.05,
                                                                            presence_penalty=True,
                                                                            max_tokens=100,
                                                                            temperature=0.3,
                                                                            top_p=0.7,
                                                                            top_k=20,
                                                                            skip_special_tokens=True))[0]['response']
            new_data['label'] = golden_label
            new_data['answer'] = response.replace('<|user|> ', '').strip()
            new_datas.append(new_data)
            # print(new_datas)
            # break
        with open(f'/data/qjw/ms-swift-main/5000other_testall/{file[:-6]}-glm4base_94400.json', 'w', encoding='utf8') as f:
            json.dump(new_datas, f, ensure_ascii=False, indent=2)
            # for item in new_datas:
            #     json_line = json.dumps(item, ensure_ascii=False)  # 将字典转换为 JSON 格式的字符串
            #     f.write(json_line + '\n')