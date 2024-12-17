import gradio as gr
from typing import Iterator
# from demo_2 import run
import re
import copy
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("/home/sda/wangzhijun/Doctor_1st/MyWorks/checkpoint-45000", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/home/sda/wangzhijun/Doctor_1st/MyWorks/checkpoint-45000",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().to(device)


ROLE_TEMP = 0
# DESCRIPTION = """
# <br>
# <div align="center" style="font-size: 13pt;">
# 🤗 <a href="https://huggingface.co/DUTIR-BioNLP/Taiyi-LLM" rel="nofollow">Hugging Face</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤖 <a href="" rel="nofollow">ModelScope</a>&nbsp;&nbsp; | &nbsp;&nbsp;<img src="https://pic.imgdb.cn/item/651401a4c458853aef46f7f5.png" style="width: 13pt; display: inline-block;"> <a href="https://github.com/DUTIR-BioNLP/Taiyi-LLM">Github</a>
# <br>
# <br>
# </div>
# <br>
# <center><font size=4>🚀<b style="color:red">Taiyi</b> is a bilingual (Chinese and English) biomedical large language model released by the Information Retrieval Laboratory of Dalian University of Technology.</font></center>
# <br>
# <center><font size=4>👆You can use the <b style="color: red">Enter</b> key to create a new line, and you can also hold down the <b style="color: red">Shift + Enter</b> keys to submit.</font></center>
# <br>
# """

BUTTON_DES = """
- <font size=3><b style="color:red">Retry</b>: If you are not satisfied with the current output of the model, you could use the Retry button to regenerate the response.</font>

- <font size=3><b style="color:red">Undo</b>: You can use the Undo button to revert the previous input.</font>

- <font size=3><b style="color:red">Clean</b>: Clear the history of the chat and start a new chat. If the topic of your conversation changes, it is recommended to use the Clean button to start a new chat.</font>

- <font size=3><b style="color:red">Submit</b>: Submit the text in the input box to the model.</font>
"""

# CONTRY_NUM = """
# <br>
# <center><a href="https://info.flagcounter.com/w63t"><img src="https://s11.flagcounter.com/mini/w63t/bg_FFFFFF/txt_000000/border_CCCCCC/flags_0/" alt="Flag Counter" border="0"></a></center>
# <center><font size=3>⚠️ Please read the <a href="https://github.com/DUTIR-BioNLP/Taiyi-LLM#disclaimer" rel="nofollow">Disclaimer</a> on Github before you strat trying Taiyi.</font></center>
# """


def check_ch_en(text):
    ch_pat = re.compile(u'[\u4e00-\u9fa5]') 
    en_pat = re.compile(r'[a-zA-Z]')
    has_ch = ch_pat.search(text)
    has_en = en_pat.search(text)
    if has_ch and has_en:
        return True 
    elif has_ch:
        return True 
    elif has_en:
        return True 
    else:
        return False

def clear_and_save_textbox(message: str):
    return '', message

def Delete_Specified_String(history):

    # 创建一个新的二维列表，用于存储不包含'你的输入无效'回答的问答对
    filtered_history = []

    # 遍历原始二维列表
    for i, pair in enumerate(history):
        if i % 2 == 1:
            if pair["content"] != '您的输入无效，请重新输入，谢谢！':
                filtered_history.append(history[i-1])
                filtered_history.append(history[i])
                

    # 现在filtered_history中包含不包含'你的输入无效'回答的问答对
    
    return filtered_history

def delete_prev_fn(history):
    try:
        _ = history.pop()
        user_message = history.pop()
    except IndexError:
        message = ''
    return history, user_message["content"] or ''


def remove_continuous_duplicate_sentences(text):
    # 使用正则表达式分割文本，以句号、逗号、分号或换行符为分隔符
    sentences = re.split(r'([。,；，\n])', text)
    
    # 初始化一个新的文本列表，用于存储去除连续重复句子后的结果
    new_sentences = [sentences[0]]  # 将第一个句子添加到列表中
    
    # 遍历句子列表，仅添加不与前一个句子相同的句子
    for i in range(2, len(sentences), 2):
        if sentences[i] != sentences[i - 2]:
            new_sentences.append(sentences[i - 1] + sentences[i])
    
    # 重新构建文本，使用原始标点符号连接句子
    new_text = ''.join(new_sentences)
    
    return new_text


def generate(
    message: str,
    history,
    max_new_tokens: int,
    temperature: float,
    top_p: float
):
    # 检查是否为空字符
    if not check_ch_en(message):
        generator = "您的输入无效，请重新输入，谢谢！"
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": generator}]

    history_list = []
    for i in range(0, len(history), 2):  # 每次跳两步
        user_message = history[i]["content"]
        assistant_message = history[i + 1]["content"]
        # 将每对 [message, generator] 加入到新列表
        history_list.append([user_message, assistant_message])

    request_list = [{'query': message, 'history':history_list}]
    response_iterator = inference_stream_vllm(
                llm_engine,
                template,
                request_list,
                generation_config=VllmGenerationConfig(
                    repetition_penalty=1.05,
                    presence_penalty=True,
                    max_tokens=500,
                    temperature=0.3,
                    top_p=0.7,
                    top_k=20,
                    skip_special_tokens=True,
                    stop_token_ids=[151329, 151336, 151338]
                )
            )
    history = Delete_Specified_String(history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ''})
    for response_batch in response_iterator:
        for response in response_batch:
            # Print the answer
            answer = response.get('response', '')
            history[-1]['content'] = answer
            # time.sleep(0.05)
            yield history



# def change_textbox(choice):
#     global ROLE_TEMP
#     #根据不同输入对输出控件进行更新
#     if choice == "中文":
#         ROLE_TEMP = 1
#         role_state = gr.update(visible=True, value="请扮演一名医疗专家，回答后续问题。")

#         message = role_state['value']
#         history = []
#         response = generate(message,history,500, 0.10, 0.9)
#         return response
#     elif choice == "English":
#         ROLE_TEMP = 2
#         role_state = gr.update(visible=True, value="Act as a medical expert to answer the questions.")

#         message = role_state['value']
#         history = []
#         response = generate(message,history,500, 0.10, 0.9)
#         return response
#     else:
#         ROLE_TEMP = 0
#         message = ""
#         history = []
#         return []


# def change_default_text():
#     return textbox.update(placeholder='Type a message...')


# def role_process():
#     if ROLE_TEMP == 1:
#         message = "请扮演一名医疗专家，回答后续问题。"
#         history = []
#         response = generate(message,history,500, 0.10, 0.9)
#         return response
#     elif ROLE_TEMP == 2:
#         message = "Act as a medical expert to answer the questions."
#         history = []
#         response = generate(message,history,500, 0.10, 0.9)
#         return response
#     elif ROLE_TEMP == 0:
#         message = ""
#         history = []
#         return []




# 处理示例问题的函数
def process_example(message: str):
    generator = generate(message, [], 2048, 0.30, 0.7)
    
    return '', generator


custom_css = """
#banner-image {
    margin-left: auto;
    margin-right: auto;
    width: 65%;
    height: 65%
}

#

"""
with gr.Blocks(css=custom_css) as demo:
    
    gr.Image("/DUTIR/Taiyi2-test/demo/logo.png", elem_id="banner-image", show_label=False, container=False)
    # with gr.Column():
    #     gr.Markdown(DESCRIPTION)
    
    with gr.Accordion(label = '⚠️ - About these buttons', open = False):
        gr.Markdown(BUTTON_DES)

    with gr.Group():
        chatbot = gr.Chatbot(
            label = 'Chatbot',
            type='messages',
            avatar_images=("/DUTIR/Taiyi2-test/demo/yonghu.png", "/DUTIR/Taiyi2-test/demo/jiqiren.png"),
            bubble_full_width=True    
        )
        
        # radio = gr.Radio(
        # ["中文", "English","None"],
        # value="None",
        # label="RolePlay(角色扮演)"
        # )
        textbox = gr.Textbox(
            container=False,
            show_label=False,
            placeholder="Please enter content...",
            lines=6
        )

        # radio.change(fn=change_textbox, inputs=radio, outputs=chatbot)
        # radio.change(fn=change_default_text, outputs=textbox)

                
    with gr.Row():
        retry_button = gr.Button('🔄  Retry', variant='secondary')
        undo_button = gr.Button('↩️ Undo', variant='secondary')
        clear_button = gr.Button('🗑️  Clear', variant='secondary')
        submit_button = gr.Button('🚩 Submit',variant='primary')


    saved_input = gr.State()

    with gr.Accordion(label = 'Advanced options', open = False):
        max_new_tokens = gr.Slider(
            label='Max new tokens',
            minimum=1,
            maximum=3000,
            step=1,
            value=2048,
            interactive=True,
        )
        temperature = gr.Slider(
            label='Temperature',
            minimum=0,
            maximum=1,
            step=0.05,
            value=0.30,
            interactive=True,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0,
            maximum=1.0,
            step=0.1,
            value=0.7,
            interactive=True,
        )



    with gr.Accordion(label = 'Examples', open = False):
        gr.Examples(
        examples=[
            '最近肚子总是隐隐作痛，感觉胀胀的，吃下去的东西都没法吸收，胃疼的特别厉害，偶尔伴有恶心想吐的感觉，请问是什么回事？',
            'What is the best treatment for sleep problems?'
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Question Answering'
        )

        gr.Examples(
        examples=[
            "患者：小孩受凉了，流清鼻涕，咳嗽，应该是风寒咳嗽，去药店买哪种药好呢\n医生：你好，宝宝咳嗽，流涕比较常见，西医角度上呼吸道感染可能性大，中医上叫做风寒咳嗽，请问宝宝除了咳嗽有没有其他不适症状呢？例如发热等，请详细描述一下，我好帮你诊治分析病情\n患者：精神状态好，也没有发热，就是喉咙有一点痛，咳嗽\n医生：先帮你分析一下病情，宝宝受凉之后免疫力降低，就会被细菌或病毒侵袭体内，气道分泌物增多，支气管平滑肌痉挛，咳嗽，咳痰，咽通。\n医生：目前没有发热，宝宝病情不重，不用过分紧张的。\n医生：我帮推荐治疗方法\n医生：宝宝目前多大了？有没有再医院看过？做过化验检查\n患者：嗯\n患者：7岁，没去医院，做过很多检查，平常就是爱咳嗽，喉哝发炎\n患者：医生说，扁桃体偏大\n医生：近期这次有没有去医院看过？做过检查\n医生：如果宝宝没有其他不适？可以口服氨溴索，桔贝合剂效果好\n医生：另外如果条件允许，可以做做雾化吸入治疗直接作用与支气管粘膜，效果更直接\n患者：不用做雾化吧，吃点药就行了\n医生：也可以先吃药\n患者：近期没有去过\n医生：你们这次没有去医院看过？\n患者：要吃消炎的吗\n患者：没\n患者：要吃消炎药吗\n医生：你好，可以先不吃的\n患者：那家里有蒲地蓝，可以吃吗\n患者：口服液\n患者：喉哝痛要吃吗\n医生：先治疗看看，可以吃的，假如宝宝出现发热或咳嗽加重，医院就诊，复查血常规和胸片，那个时候再考虑加抗生素\n患者：另外买个止咳的，行吗\n医生：我们的观点是宝宝小，尽量少吃消炎药，可以先吃那几个药三天看看效果\n患者：嗯谢谢\n根据上述对话，给出诊疗报告\n说明：诊疗报告分为主诉, 现病史, 辅助检查, 既往史, 诊断, 建议这六个章节。"
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Medical Report Generation'
        )

        gr.Examples(
        examples=[
            '从下面文本中识别出指定的实体类型：\n治疗以选用大环内酯类抗生素，沙眼衣原体肺炎也可用磺胺二甲基异唑，年长儿和成人用氟喹诺酮类效果也很好。\n实体类型：疾病，药物',
            'Extract the gene, disease entities from the following text:\nIdentification of a novel FBN1 gene mutation in a Chinese family with Marfan syndrome.'
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Name Entity Recognition'
        )

        gr.Examples(
        examples=[
            "给出句子中药物治疗关系类型的实体对：慢性阻塞性肺疾病@减少急性加重：有高质量的证据证实，β2 受体激动剂在减少 12-52 周急性加重方面比安慰剂更有效。",
            "Find the relations of drug entity pairs in the text：\nMitotane has been reported to accelerate the metabolism of warfarin by the mechanism of hepatic microsomal enzyme induction, leading to an increase in dosage requirements for warfarin. Therefore, physicians should closely monitor patients for a change in anticoagulant dosage requirements when administering Mitotane to patients on coumarin-type anticoagulants. In addition, Mitotane should be given with caution to patients receiving other drugs susceptible to the influence of hepatic enzyme induction.\nRelation Types: ADVISE, MECHANISM, EFFECT, INT"
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Relation Extraction'
        )

        gr.Examples(
        examples=[
            "找出指定的临床发现事件属性：\n因患者需期末考试，故予以口服“雷贝拉唑钠肠溶片”治疗，现腹痛情况明显好转。\n事件抽取说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成",
            'Input text: "Contaminated drinking water is responsible for causing diarrheal diseases that kill millions of people a year.\nEven Types: Treatment of disease, Cause of disease\nRole Types: Cause, Theme\nPlease extract events from the input text.'
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Event Extraction'
        )

        gr.Examples(
        examples=[
            "将下面中文文本翻译成英文：\n光动力疗法（PDT）作为一种新兴的肿瘤治疗手段，因其不良反应较少、靶向性好、可重复治疗等优点，已广泛应用于临床多种肿瘤的治疗。相比于手术、化疗及放疗等传统治疗策略，光动力疗法不仅可杀伤原位肿瘤，还可通过激活机体的免疫效应对转移瘤发挥抑制作用。然而，PDT诱导免疫效应的高低受多种因素影响，包括光敏剂在细胞内的定位和剂量、光参数、肿瘤内的氧浓度、免疫功能的完整性等。本文针对PDT介导抗肿瘤免疫效应的相关机制，以及PDT免疫效应的主要影响因素进行综述，以探讨PDT用于肿瘤治疗的未来发展方向。",
            "Translate the following text into Chinese:\nNon-Alcoholic Fatty Liver Disease (NAFLD) is defined as increased liver fat percentage, and is the most common chronic liver disease in children. Rather than NAFLD, Metabolic-Associated Fatty Liver Disease (MAFLD), defined as increased liver fat with presence of adverse cardio-metabolic measures, might have more clinical relevance in children. We assessed the prevalence, risk-factors and cardio-metabolic outcomes of MAFLD at school-age."
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Machine Translation'
        )

        gr.Examples(
        examples=[
            "请给下面摘要起标题：\n气管食管瘘是指气管或支气管与食管之间的病理性瘘道，包括气管-食管瘘和支气管-食管瘘，临床以气管-食管瘘较多见。气管食管瘘致病原因较多，可引起严重的并发症，是对患者生活质量影响显著、治疗困难和病死率较高的疾病。气管食管瘘目前治疗方式较多，但多数疗效欠佳，对新兴治疗手段的需求迫切。胸腹部X线摄影检出鼻胃管滞留是气管食管瘘诊断的金标准，其主要治疗方法包括外科手术治疗、支架置入、局部生物胶水封闭、干细胞治疗等。本文综述近年气管食管瘘诊断与治疗的主要研究进展，旨在为该病的临床诊治提供参考。",
            "Output a title for the following abstract:\nThe incidence of diabetes mellitus has been increasing, prompting the search for non-invasive diagnostic methods. Although current methods exist, these have certain limitations, such as low reliability and accuracy, difficulty in individual patient adjustment, and discomfort during use. This paper presents a novel approach for diagnosing diabetes using high-frequency ultrasound (HFU) and a convolutional neural network (CNN). This method is based on the observation that glucose in red blood cells (RBCs) forms glycated hemoglobin (HbA1c) and accumulates on its surface. The study incubated RBCs with different glucose concentrations, collected acoustic reflection signals from them using a custom-designed 90-MHz transducer, and analyzed the signals using a CNN. The CNN was applied to the frequency spectra and spectrograms of the signal to identify correlations between changes in RBC properties owing to glucose concentration and signal features. The results confirmed the efficacy of the CNN-based approach with a classification accuracy of 0.98. This non-invasive diagnostic technology using HFU and CNN holds promise for in vivo diagnosis without the need for blood collection."
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Title Generation'
        )

        gr.Examples(
        examples=[
            "现有以下文本：\n治皮肤病费用大概多少？\n请将上述文本分类至指定类别中：医疗费用，后果表述，指标解读，病情诊断，就医建议，疾病描述，其他，治疗方案，病因分析，功效作用，注意事项",
            'Document triage: "Will my mask from sherwin williams paint store with filters protect me from corona virus along with paint fumes?"\nLabels: patient, doctor'
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Text Classification'
        )

        gr.Examples(
        examples=[
            "语句1：乙肝小二阳会转成小三阳吗？\n语句2：乙肝小三阳会不会转成肝硬化、肝癌？\n请从下面选项中评估这段文本的语义相似度：语义不相同，语义相同",
            "1. How can someone's happiness level affect someone's health?\n2. Can staying happy improve my health? What specific steps should I take?\nAssess the semantic similarity of the text pairs based on the following labels: dissimilar, similar"
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=False,
        label='Text Semantic Similarity'
        )


    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
        ],
        outputs=chatbot,
        api_name=False,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )


    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

    # gr.HTML(CONTRY_NUM)


if __name__ == "__main__":
    demo.queue().launch()
