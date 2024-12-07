import json 
import os
from collections import defaultdict


gold_path = '/data/qjw/ms-swift-main/test_data/QA'
pred_path = '/data/qjw/ms-swift-main/5000other_testall'

for dataset_file in os.listdir(gold_path):
        # print(dataset_file)
    if dataset_file.startswith('QA_choice'):
        true_answer = gold_path+'/'+dataset_file
        with open(true_answer, 'r', encoding='utf-8') as file:
            trueAnswers = []
            
            for line in file:
                data = json.loads(line)
                trueAnswer = data["conversation"][0]["assistant"]
        
                # 英文数据集则将此处换为英文冒号":"，中文数据集则换成中文冒号"："
                if dataset_file.find('_zh')>=0:
                    trueAnswer = trueAnswer.split("：")[1].strip()
                elif dataset_file.find('_en')>=0:
                    trueAnswer = trueAnswer.split(":")[1].strip()
        
                trueAnswers.append(trueAnswer)
        
        # 英文数据集则将此处换为 answer_flag = "Answer:"，中文数据集则换成 answer_flag = "答案："
        # answer_flag = "答案："
        model_answer = pred_path+'/'+dataset_file[:-6]+'-glm4base_94400.json'
        with open(model_answer, 'r', encoding='utf-8') as f:
            modelAnswers = []
            for line in f:
                data = json.loads(line)
                modelAnswer = data["answer"]
                # print(modelAnswer)
                # if answer_flag in modelAnswer:
        
                    # 英文数据集则将此处换为英文冒号":"，中文数据集则换成中文冒号"："
                if dataset_file.find('_zh')>=0:
                    if '答案：' in modelAnswer:
                        modelAnswer = modelAnswer.split("：")[1].strip()

                elif dataset_file.find('_en')>=0:
                    if 'Answer:' in modelAnswer:
                        modelAnswer = modelAnswer.split(":")[1].strip()                           
    
                modelAnswers.append(modelAnswer)
                # else:
                #     modelAnswers.append(modelAnswer)
        
            # print(modelAnswers)
        num = 0
        Total_num = 0
        for i in range(len(modelAnswers)):
            Total_num += 1
            if modelAnswers[i] == trueAnswers[i]:
                num += 1
        
        print(dataset_file)
        print("准确率为：" + str(num/Total_num), num, Total_num)



