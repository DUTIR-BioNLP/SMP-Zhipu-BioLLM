import json 
import os
from collections import defaultdict

point_id = '22400'
folder_path = 'glm4-9b-chat/v6-20240924-223358'

'''RE-bc5cdr-test'''
file_path = f'/data/qjw/ms-swift-main/5000other_testall/RE-bc5cdr-test-glm4base_94400.json'
with open(file_path, 'r', encoding='utf8') as f:
    datas = json.load(f)
    gold_item_num_total = 0
    pred_ture_item_num_total = 0
    pred_item_num_total = 0
    for data in datas:
        if ':' in data['label']:
            label = data['label'].lower().split(':')
            if ';' in label[1]:
                goldens = label[1].split(';')
                for idx, golden in enumerate(goldens):
                    goldens[idx] = golden.strip()
                goldens = list(set(goldens))
            else:
                goldens = [label[1].strip()]
        else:
            goldens = []
        goldens_num = len(goldens)

        new_goldens = []
        for golden in goldens:
            try:
                A, B = golden[1:-1].split(', ')
                new_golden = '[{}, {}]'.format(B, A)
                new_goldens.append(new_golden)
            except:
                pass
        goldens += new_goldens


        if ':' in data['answer']:
            answer = data['answer'].lower().split(':')
            if ';' in answer[1]:
                preds = answer[1].split(';')
                for idx, pred in enumerate(preds):
                    preds[idx] = pred.strip()                
                preds = list(set(preds))
            else:
                preds = [answer[1].strip()]
        else:
            preds = []
        
        gold_item_num_total += goldens_num
        pred_item_num_total += len(preds)
        for pred in preds:
            if pred in goldens:
                pred_ture_item_num_total += 1
    print('\nCDR:')
    print("标签中一共有{}个关系二元组".format(gold_item_num_total))
    print("模型共预测了{}个关系二元组".format(pred_item_num_total))
    print("其中{}个正确的".format(pred_ture_item_num_total))
    print("P:",pred_ture_item_num_total/pred_item_num_total)
    print("R:",pred_ture_item_num_total/gold_item_num_total)
    print("F1:",2/(1/(pred_ture_item_num_total/pred_item_num_total)+1/(pred_ture_item_num_total/gold_item_num_total)))

'''RE-biorelex-dev'''
file_path = f'/data/qjw/ms-swift-main/5000other_testall/RE-biorelex-dev-glm4base_94400.json'
with open(file_path, 'r', encoding='utf8') as f:
    datas = json.load(f)
    gold_item_num_total = 0
    pred_ture_item_num_total = 0
    pred_item_num_total = 0
    for data in datas:
        if ':' in data['label']:
            label = data['label'].lower().split(':')
            if ';' in label[1]:
                goldens = label[1].split(';')
                for idx, golden in enumerate(goldens):
                    goldens[idx] = golden.strip()
                goldens = list(set(goldens))
            else:
                goldens = [label[1].strip()]
        else:
            goldens = []
        goldens_num = len(goldens)

        new_goldens = []
        for golden in goldens:
            try:
                A, B = golden[1:-1].split(', ')
                new_golden = '[{}, {}]'.format(B, A)
                new_goldens.append(new_golden)
            except:
                pass
        goldens += new_goldens        

        if ':' in data['answer']:
            answer = data['answer'].lower().split(':')
            if ';' in answer[1]:
                preds = answer[1].split(';')
                for idx, pred in enumerate(preds):
                    preds[idx] = pred.strip()                
                preds = list(set(preds))
            else:
                preds = [answer[1].strip()]
        else:
            preds = []
        
        gold_item_num_total += goldens_num
        pred_item_num_total += len(preds)
        for pred in preds:
            if pred in goldens:
                pred_ture_item_num_total += 1
    print('\nbiorelex:')
    print("标签中一共有{}个关系二元组".format(gold_item_num_total))
    print("模型共预测了{}个关系二元组".format(pred_item_num_total))
    print("其中{}个正确的".format(pred_ture_item_num_total))
    print("P:",pred_ture_item_num_total/pred_item_num_total)
    print("R:",pred_ture_item_num_total/gold_item_num_total)
    print("F1:",2/(1/(pred_ture_item_num_total/pred_item_num_total)+1/(pred_ture_item_num_total/gold_item_num_total)))


'''RE-cmeie_v2-dev'''
file_path = f'/data/qjw/ms-swift-main/5000other_testall/RE-cmeie_v2-dev-glm4base_94400.json'
with open(file_path, 'r', encoding='utf8') as f:
    datas = json.load(f)
    gold_item_num_total = 0
    pred_ture_item_num_total = 0
    pred_item_num_total = 0
    for data in datas:
        labels_num = 0
        if data['label'] != '上述文本中没有相关的实体关系。':
            types = data['label'].lower().split('\n')
            goldens = {}
            for type in types:
                if type != '':    
                    name = type.strip().split('：')[0]
                    value = type.strip().split('：')[1]
                    if ';' in value:
                        label_ori = value.split(';')
                        labels = []
                        for item in label_ori:
                            labels.append(item.strip())
                        labels = list(set(labels))
                    else:
                        labels = [value.strip()]
                    labels_num += len(labels)

                    new_labels = []
                    for label in labels:
                        try:
                            A, B = label[1:-1].split(', ')
                            new_label = '[{}, {}]'.format(B, A)
                            new_labels.append(new_label)
                        except:
                            pass
                    labels += new_labels      
                    goldens[name] = labels
        else:
            goldens = {}

        types = data['answer'].lower().split('\n')
        preds = {}
        for type in types:
            if (type != '') and ('：' in type):    
                name = type.split('：')[0]
                value = type.split('：')[1]
                if ';' in value:
                    label_ori = value.split(';')
                    label = []
                    for item in label_ori:
                        label.append(item.strip())
                    label = list(set(label))
                else:
                    label = [value.strip()]
                preds[name] = label
        
        goldens_len = labels_num
        preds_len = sum([len(value) for value in preds.values()])
        gold_item_num_total += goldens_len
        pred_item_num_total += preds_len
        for key, value in preds.items():
            if key in goldens.keys():
                for label in value:
                    if label in goldens[key]:
                        pred_ture_item_num_total += 1
    print('\ncmeie:')
    print("标签中一共有{}个关系二元组".format(gold_item_num_total))
    print("模型共预测了{}个关系二元组".format(pred_item_num_total))
    print("其中{}个正确的".format(pred_ture_item_num_total))
    print("P:",pred_ture_item_num_total/pred_item_num_total)
    print("R:",pred_ture_item_num_total/gold_item_num_total)
    print("F1:",2/(1/(pred_ture_item_num_total/pred_item_num_total)+1/(pred_ture_item_num_total/gold_item_num_total)))

'''RE-ddi_corpus-test'''
file_path = f'/data/qjw/ms-swift-main/5000other_testall/RE-ddi_corpus-test-glm4base_94400.json'
with open(file_path, 'r', encoding='utf8') as f:
    datas = json.load(f)
    gold_item_num_total = 0
    pred_ture_item_num_total = 0
    pred_item_num_total = 0
    for data in datas:
        labels_num = 0
        if data['label'] != 'The text does not contain any specified relations.':
            types = data['label'].lower().split('\n')
            goldens = {}
            for type in types:
                if type != '':    
                    name = type.strip().split(':')[0]
                    value = type.strip().split(':')[1]
                    if ';' in value:
                        label_ori = value.split(';')
                        labels = []
                        for item in label_ori:
                            labels.append(item.strip())
                        labels = list(set(labels)) 
                    else:
                        labels = [value.strip()]
                    labels_num += len(labels)

                    new_labels = []
                    for label in labels:
                        try:
                            A, B = label[1:-1].split(', ')
                            new_label = '[{}, {}]'.format(B, A)
                            new_labels.append(new_label)
                        except:
                            pass
                    labels += new_labels      
                    
                    goldens[name] = labels
        else:
            goldens = {}
        
        if ':' in data['answer']:
            types = data['answer'].lower().split('\n')
            preds = {}
            for type in types:
                if (type != '') and (':' in type):    
                    name = type.split(':')[0]
                    value = type.split(':')[1]
                    if ';' in value:
                        label_ori = value.split(';')
                        label = []
                        for item in label_ori:
                            label.append(item.strip())
                        label = list(set(label))
                    else:
                        label = [value.strip()]
                    preds[name] = label
        else:
            preds = {}
        
        goldens_len = labels_num
        preds_len = sum([len(value) for value in preds.values()])
        gold_item_num_total += goldens_len
        pred_item_num_total += preds_len
        for key, value in preds.items():
            if key in goldens.keys():
                for label in value:
                    if label in goldens[key]:
                        pred_ture_item_num_total += 1
    print('\nddi:')
    print("标签中一共有{}个关系二元组".format(gold_item_num_total))
    print("模型共预测了{}个关系二元组".format(pred_item_num_total))
    print("其中{}个正确的".format(pred_ture_item_num_total))
    print("P:",pred_ture_item_num_total/pred_item_num_total)
    print("R:",pred_ture_item_num_total/gold_item_num_total)
    print("F1:",2/(1/(pred_ture_item_num_total/pred_item_num_total)+1/(pred_ture_item_num_total/gold_item_num_total)))
