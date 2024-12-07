# 训练框架

我们主要使用ms-swift框架进行模型的训练，链接如下[https://github.com/modelscope/ms-swift?tab=readme-ov-file]

## 安装

本项目Python版本为3.11

通过源代码安装SWIFT后，还需安装vllm库，以便进行推理，请在安装SWIFT后运行以下命令：

```
pip install vllm==0.6.0
```

具体配置信息请参考requirements.txt文件



# 模型训练

本项目使用8张A5000进行glm4-9b模型的全参量微调，使用以下脚本进行模型训练：

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft 
				--sft_type 'full' 
                --dtype 'bf16' 
                --model_id_or_path './ZhipuAI/glm-4-9b' 
                --template_type 'chatglm4' 
                --dataset './swift/data/your_dataset.jsonl' 
                --lora_target_modules 'ALL' 
                --init_lora_weights 'True' 
                --learning_rate '1e-05' 
                --num_train_epochs '4' 
                --gradient_accumulation_steps '16' 
                --eval_steps '25000' 
                --save_steps '25000' 
                --eval_batch_size '1' 
                --model_type 'glm4-9b'  
                --add_output_dir_suffix False 
                --output_dir ./swift/output/glm4-9b/v2-20241204-215135
                --logging_dir ./swift/output/glm4-9b/v2-20241204-215135/runs 
                --ignore_args_error True
                --save_total_limit 9
```



# 模型测试

对于本项目，可以使用以下命令在终端中进行模型问答:
```python 
python testCHAT.py
```

对于本项目使用的测试集，主要使用以下命令分别进行模型测试：

```python
cd test_file
python testNER.py
python testRE.py
python testTC.py
python testQA.py
```

对于测试结果的评价，主要使用以下命令分别进行结果的评估：

```
cd test_file
python eval_NER.py
python eval_RE.py
python eval_TC.py
python eval_QA.py
```

