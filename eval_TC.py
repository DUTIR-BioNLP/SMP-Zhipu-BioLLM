import json 
import os
from collections import defaultdict
import tqdm
# gold_path = './文本分类/English/bc7_litcovid/bc7_t.jsonl'
# pred_path = './newnew_bc7.jsonl'

def TC_eval(gold_path,pred_path):
        
    gold_result,pred_result = defaultdict(list),defaultdict(list)
    
    for dataset_file in os.listdir(gold_path):
        # print(dataset_file)
        if dataset_file.startswith('TC-') or dataset_file.startswith('TP_ss'):
            file_path = gold_path + '/' + dataset_file
            # print(file_path)
            gold_result[dataset_file] = []
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    gold_result[dataset_file].append(data['conversation'][0]['assistant'])
        
    
    for dataset_file in os.listdir(pred_path):
        if '-glm4base_94400' in dataset_file and dataset_file.startswith('TC-'): #if dataset_file.find('118400') and dataset_file.startswith('TC-') or dataset_file.startswith('TP_ss'):
            print(dataset_file)
            file_path = pred_path + '/' + dataset_file
            # print(file_path)
            pred_result[dataset_file] = []
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    pred_result[dataset_file.replace('-glm4base_94400','')].append(data["answer"])      # 修改前：pred_result[dataset_file.replace('_answer','')].append(data["answer"])
    
    header = defaultdict(list)
    header['TC-chip_ctc-dev.jsonl'] = '伦理审查，疾病，吸烟状况，预期寿命，依存性，肿瘤进展，受体状态，过敏耐受，实验室检查，年龄，性别，教育情况，研究者决定，健康群体，知情同意，酒精使用，体征(医生检测），口腔相关，药物，参与其它试验，器官组织状态，风险评估，锻炼，设备，护理，成瘾行为，读写能力，性取向，症状(患者感受)，献血，病例来源，数据可及性，特殊病人特征，怀孕相关，睡眠，治疗或手术，能力，饮食，残疾群体，种族，含有多类别的语句，居住情况，诊断，疾病分期'
    header['TC-bc7_litcovid-test.jsonl'] = "Case Report, Prevention, Transmission, Diagnosis, Mechanism, Treatment, Epidemic Forecasting"
    header['TC-kuake_qic-dev.jsonl'] = "医疗费用，后果表述，指标解读，病情诊断，就医建议，疾病描述，其他，治疗方案，病因分析，功效作用，注意事项"
    header['TC-meddialog-test.jsonl'] = "patient, doctor"
    header['TC-hallmarks_of_cancer-test.jsonl'] = "evading growth suppressors, tumor promoting inflammation, enabling replicative immortality, cellular energetics, resisting cell death, activating invasion and metastasis, genomic instability and mutation, none, inducing angiogenesis, sustaining proliferative signaling, avoiding immune destruction"
    header['TP_ss-biosses-test.jsonl'] = "different topics, not equivalent but are on the same topic, not equivalent but share some details, roughly equivalent but some important information differs/missing, completely or mostly equivalent"
    header['TP_ss-chip_sts-dev.jsonl'] = "语义不相同，语义相同"
    
    for k,v in header.items():
        if k in ['TC-bc7_litcovid-test.jsonl','TC-meddialog-test.jsonl','TP_ss-biosses-test.jsonl','TC-hallmarks_of_cancer-test.jsonl']:
            v_l = v.split(',')
        else:
            v_l = v.split('，')
        v_d = dict()
        for i, v_c in enumerate(v_l):
            v_d[v_c.strip()] = i
        header[k] = v_d
        
    # 生成结果
    from sklearn.metrics import classification_report
    for dataset_file in os.listdir(gold_path):
        if dataset_file in ['TP_ss-chip_sts-dev.jsonl']:
            print(dataset_file)
            sum = 0 
            right, all_ = 0,0
            g,p = 0,0
            labelgs,predgs = [],[]
    
            for id, (pred, gold) in enumerate(zip(pred_result[dataset_file], gold_result[dataset_file])):
            
                # print(pred.strip().split(':')[-1].strip())
                if pred.strip().split(':')[-1].strip() in header[dataset_file].keys():
                    sum += 1
                gold = gold.strip().split(';')
                pred = pred.strip().split(';')
                # print(gold,pred)
                # right, all_ = 0,0
                # g,p = 0,0
                # if gold[0] == pred[0]:
                #     print(id)
                right += len(set(gold).intersection(set(pred)))
                all_ += len(set(gold).union(set(pred)))
                g += len(gold)
                p += len(pred)
    
                label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                # print(gold[0],pred[0])
                if pred[0].strip() == '':
                    for ki in header[dataset_file].keys():
                        if ki != gold[0]:
                            pred[0] = ki
                            break
                if pred[0] not in header[dataset_file].keys():
                    for ki in header[dataset_file].keys():
                        if ki != gold[0]:
                            pred[0] = ki
                            break
    
                label_g[header[dataset_file][gold[0]]] = 1
                label_p[header[dataset_file][pred[0]]] = 1
                predgs.append(label_p)
                labelgs.append(label_g)
    
            print(sum)
            a = right/p
            b = right/g
            f = 2*a*b/(a+b)
            acc = right/all_
            print(a,b,f,acc)
            print(classification_report(labelgs, predgs, digits=4,
                                    target_names=header[dataset_file].keys()))
            
        elif dataset_file in ['TP_ss-biosses-test.jsonl']:
            print(dataset_file)
            sum = 0 
            right, all_ = 0,0
            g,p = 0,0
            labelgs,predgs = [],[]
    
            for id, (pred, gold) in enumerate(zip(pred_result[dataset_file], gold_result[dataset_file])):
            
                # print(pred.strip().split(':')[-1].strip())
                if pred.strip().split(':')[-1].strip() in header[dataset_file].keys():
                    sum += 1
                gold = gold.strip().split(';')
                pred = pred.strip().split(';')
                # print(gold,pred)
                # right, all_ = 0,0
                # g,p = 0,0
                # if gold[0] == pred[0]:
                #     print(id)
                right += len(set(gold).intersection(set(pred)))
                all_ += len(set(gold).union(set(pred)))
                g += len(gold)
                p += len(pred)
    
                label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                # print(gold[0],pred[0])
                if pred[0].strip() == '':
                    for ki in header[dataset_file].keys():
                        if ki != gold[0]:
                            pred[0] = ki
                            break
                if pred[0] not in header[dataset_file].keys():
                    for ki in header[dataset_file].keys():
                        if ki != gold[0]:
                            pred[0] = ki
                            break
    
                label_g[header[dataset_file][gold[0]]] = 1
                label_p[header[dataset_file][pred[0]]] = 1
                predgs.append(label_p)
                labelgs.append(label_g)
    
            print(sum)
            a = right/p
            b = right/g
            f = 2*a*b/(a+b)
            acc = right/all_
            print(a,b,f,acc)
            print(classification_report(labelgs, predgs, digits=4,
                                    target_names=header[dataset_file].keys()))
        elif dataset_file in ['TC-kuake_qic-dev.jsonl','TC-chip_ctc-dev.jsonl']:
            print(dataset_file)
            sum = 0 
            right, all_ = 0,0
            g,p = 0,0
            labelgs,predgs = [],[]
    
            for id, (pred, gold) in enumerate(zip(pred_result[dataset_file], gold_result[dataset_file])):
                if 'Result:' in pred or '上述文本被分类为' in pred or '上述文本对应的类别是' in pred or '上述文本的分类结果为' in pred:
                    sum+=1
                    gold = gold.strip().split('：')[-1].replace(' ','').replace('疾病表述','疾病描述').split(';')
                    pred = pred.strip().split('：')[-1].replace(' ','').replace('疾病表述','疾病描述').split(';')
                    # gold = [g.strip() for g in gold.strip().split(':')[-1].strip().split(';')]
                    # pred = [p.strip() for p in pred.strip().split(':')[-1].strip().split(';')]
                    # print(gold,pred,id)
                    # print(id)
    
                    # right, all_ = 0,0
                    # g,p = 0,0
                    right += len(set(gold).intersection(set(pred)))
                    all_ += len(set(gold).union(set(pred)))
                    g += len(gold)
                    p += len(pred)
    
                    # label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                    # # print(gold[0],pred[0])
                    # if pred[0].strip() == '':
                    #     for ki in header[dataset_file].keys():
                    #         if ki != gold[0]:
                    #             pred[0] = ki
                    #             break
                    # if pred[0] not in header[dataset_file].keys():
                    #     for ki in header[dataset_file].keys():
                    #         if ki != gold[0]:
                    #             pred[0] = ki
                    #             break
    
                    # label_g[header[dataset_file][gold[0]]] = 1
                    # label_p[header[dataset_file][pred[0]]] = 1
                    # predgs.append(label_p)
                    # labelgs.append(label_g)
                    # 多标签：
                    label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                    # print(gold[0],pred[0])
                    for p_c in pred:
                        if p_c not in header[dataset_file].keys():
                            for ki in header[dataset_file].keys():
                                if ki not in gold:
                                    p_c = ki
                                    break
                        label_p[header[dataset_file][p_c]] = 1
                    predgs.append(label_p)
                    for g_c in gold:
                        label_g[header[dataset_file][g_c]] = 1
                    labelgs.append(label_g)
    
    
            print(sum)
            a = right/p
            b = right/g
            f = 2*a*b/(a+b)
            acc = right/all_
            print(a,b,f,acc)
            print(classification_report(labelgs, predgs, digits=4,
                                    target_names=header[dataset_file].keys()))
        elif dataset_file in ['TC-bc7_litcovid-test.jsonl', 'TC-meddialog-test.jsonl','TC-hallmarks_of_cancer-test.jsonl']: # elif dataset_file in ['TC-bc7_litcovid-test.jsonl','TC-meddialog-test.jsonl','TC-hallmarks_of_cancer-test.jsonl']:
            print(dataset_file)
            sum = 0 
            right, all_ = 0,0
            g,p = 0,0
            labelgs,predgs = [],[]
    
            for id, (pred, gold) in enumerate(zip(pred_result[dataset_file], gold_result[dataset_file])):
                # print(gold,id)
                if 'Result:' in pred or '上述文本被分类为' in pred or '上述文本对应的类别是' in pred or '上述文本的分类结果为' in pred:
                    sum+=1
                    # gold = gold.strip().split('：')[-1].replace(' ','').replace('疾病表述','疾病描述').split(';')
                    # pred = pred.strip().split('：')[-1].replace(' ','').replace('疾病表述','疾病描述').split(';')
                    gold = [g.strip() for g in gold.strip().split(':')[-1].strip().split(';')]
                    pred = [p.strip() for p in pred.strip().split(':')[-1].strip().split(';')]
                    # print(gold,pred,id)
                    # print(id)
    
                    # right, all_ = 0,0
                    # g,p = 0,0
                    right += len(set(gold).intersection(set(pred)))
                    all_ += len(set(gold).union(set(pred)))
                    g += len(gold)
                    p += len(pred)
    
                    # label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                    # # print(gold[0],pred[0])
                    # if pred[0].strip() == '':
                    #     for ki in header[dataset_file].keys():
                    #         if ki != gold[0]:
                    #             pred[0] = ki
                    #             break
                    # if pred[0] not in header[dataset_file].keys():
                    #     for ki in header[dataset_file].keys():
                    #         if ki != gold[0]:
                    #             pred[0] = ki
                    #             break
    
                    # label_g[header[dataset_file][gold[0]]] = 1
                    # label_p[header[dataset_file][pred[0]]] = 1
                    # predgs.append(label_p)
                    # labelgs.append(label_g)
                    # 多标签：
                    label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                    # print(gold[0],pred[0])
                    for p_c in pred:
                        if p_c not in header[dataset_file].keys():
                            for ki in header[dataset_file].keys():
                                if ki not in gold:
                                    p_c = ki
                                    break
                        label_p[header[dataset_file][p_c]] = 1
                    predgs.append(label_p)
                    # print('!!!!!')
                    # print(gold)
                    for g_c in gold:
                        label_g[header[dataset_file][g_c]] = 1
                    labelgs.append(label_g)
                    
                else:
                    continue
                    # print(pred.strip().split(':')[-1].strip())
                    if pred.strip().split(':')[-1].strip() in header[dataset_file].keys():
                        sum += 1
                    gold = gold.strip().split(';')
                    pred = pred.strip().split(';')
                    # print(gold,pred)
                    # right, all_ = 0,0
                    # g,p = 0,0
                    # if gold[0] == pred[0]:
                    #     print(id)
                    right += len(set(gold).intersection(set(pred)))
                    all_ += len(set(gold).union(set(pred)))
                    g += len(gold)
                    p += len(pred)
    
                    label_g,label_p = [0] * len(header[dataset_file]),[0] * len(header[dataset_file])
                    # print(gold[0],pred[0])
                    if pred[0].strip() == '':
                        for ki in header[dataset_file].keys():
                            if ki != gold[0]:
                                pred[0] = ki
                                break
                    if pred[0] not in header[dataset_file].keys():
                        for ki in header[dataset_file].keys():
                            if ki != gold[0]:
                                pred[0] = ki
                                break
    
                    label_g[header[dataset_file][gold[0]]] = 1
                    label_p[header[dataset_file][pred[0]]] = 1
                    predgs.append(label_p)
                    labelgs.append(label_g)
    
            print(sum)
            a = right/p
            b = right/g
            f = 2*a*b/(a+b)
            acc = right/all_
            print(a,b,f,acc)
            print(classification_report(labelgs, predgs, digits=4,
                                    target_names=header[dataset_file].keys()))



# 使用前需保证两个文件夹下的对应文件名一致            
TC_eval("/data/qjw/ms-swift-main/test_data/TC", "/data/qjw/ms-swift-main/5000other_testall")