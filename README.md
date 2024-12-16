# 模型权重

模型权重[下载地址](https://modelscope.cn/models/DUTIRbionlp/SMP-Zhipu-BioLLM-v0)

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
# 模型展示系统

修改模型路径和权重路径后，终端输入python demo.py即可本地进行测试

# 致谢
本项目工作得到中国中文信息学会社会媒体处理专委会(SMP)-智谱 AI大模型交叉学科基金资助

# 免责声明
本项目相关资源仅供学术研究之用，严禁用于商业用途。对本仓库源码的使用遵循开源许可协议 [Apache 2.0](https://github.com/DUTIR-BioNLP/Taiyi-LLM/blob/main/LICENSE)。在使用过程中，用户需认真阅读并遵守以下声明：
1. 请您确保您所输入的内容未侵害他人权益，未涉及不良信息，同时未输入与政治、暴力、色情相关的内容，且所有输入内容均合法合规。
2. 请您确认并知悉使用大模型生成的所有内容均由人工智能模型生成，生成内容具有不完全理性，本项目对其生成内容的准确性、完整性和功能性不做任何保证，亦不承担任何法律责任。
3. 本模型中出现的任何违反法律法规或公序良俗的回答，均不代表本项目的态度、观点或立场，本项目将不断完善模型回答以使其更符合社会伦理和道德规范。
4. 对于模型输出的任何内容，使用者需自行承担风险和责任，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。
5. 本项目中出现的第三方链接或库仅为提供便利而存在，其内容和观点与本项目无关。使用者在使用时需自行辨别，本项目不承担任何连带责任；
6. 若使用者发现项目出现任何重大错误，请向我方反馈，以助于我们及时修复。

使用本项目即表示您已经仔细阅读、理解并同意遵守以上免责声明。本项目保留在不预先通知任何人的情况下修改本声明的权利。
