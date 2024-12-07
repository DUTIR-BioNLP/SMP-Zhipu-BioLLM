import json 
import os
from collections import defaultdict

class NER_eval():
    def __init__(self,gold_path,pred_path):
        self.temp_path = '/data/qjw/ms-swift-main/test_result/NER/tmp'
        self.gold_path = gold_path
        self.pred_path = pred_path
    
    def com_prf(self, test_data_dir,file_name):
        count_non_json = 0
        gold_item_num_total = 0
        pred_ture_item_num_total = 0
        pred_item_num_total = 0
        oupfile = open(file_name, 'r', encoding='utf-8')
    
        oupfilelist = oupfile.readlines()
        all_gold_label = defaultdict(set)
        all_pred_label = defaultdict(set)
        all_gold = []
        all_pred = []
    
        with open(test_data_dir, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                gold_label_item = []
                pred_item = []
                line_dic = json.loads(line)
                # print('line_dic',type(line_dic),line_dic)
                # print('line_dic["conversation"][0]["assistant"]',type(line_dic["conversation"][0]["assistant"]),line_dic["conversation"][0]["assistant"])
                label_dic = line_dic["conversation"][0]["assistant"]
                # print('idx',idx)
                # print('label_dic',label_dic)
                # print('oupfilelist[idx]',json.loads(oupfilelist[idx])["answer"])
                dic_str_pre = json.loads(oupfilelist[idx])["answer"]
                # print('label_pre',dic_str_pre)
                # print("##############")
    
                for key in label_dic.keys():
                    if key in all_gold_label.keys():
                        pass
                    else:
                        all_gold_label[key] = []
                    for value in label_dic[key]:
                        all_gold_label[key].append(value+key+'idx'+str(idx))
                        all_gold.append(value+key+'idx'+str(idx))
    
                try:
                    pred_dic_item = dic_str_pre
                    # print('pred_dic_item',pred_dic_item)
                    pred_ture_item_num = 0
                    for pre_key in pred_dic_item.keys():
                        if pre_key in all_pred_label.keys():
                            pass
                        else:
                            all_pred_label[pre_key] = []
                        for pre_value in pred_dic_item[pre_key]:
                            all_pred_label[pre_key].append(pre_value + pre_key+'idx'+str(idx))
                            all_pred.append(pre_value + pre_key+'idx'+str(idx))
                except:
                    count_non_json += 1
    
        # print('all_gold_label',all_gold_label)
        # print('all_pred_label',all_pred_label)
    
        print('type_name','\t','p', '\t','r', '\t','f1')
        for type_name, true_entities in all_gold_label.items():
            # print('true_entities', true_entities)
            pred_entities = all_pred_label[type_name]
            # print('pred_entities', pred_entities)
            nb_correct = len(set(true_entities) & set(pred_entities))
            nb_pred = len(set(pred_entities))
            nb_true = len(set(true_entities))
    
            p = nb_correct / nb_pred if nb_pred > 0 else 0
            r = nb_correct / nb_true if nb_true > 0 else 0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
            print(type_name,'\t',p, '\t',r, '\t',f1)
            # print(type_name, '\t', nb_true, '\t', nb_pred)
        # print("###########"*10)
        all_gold = []
        for i in all_gold_label:
            all_gold = all_gold+all_gold_label[i]
    
        all_pred = []
        for i in all_pred_label:
            all_pred = all_pred + all_pred_label[i]
    
        nb_correct = len(set(all_gold) & set(all_pred))
        nb_pred = len(set(all_pred))
        nb_true = len(set(all_gold))
    
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
        print('all','\t',p, '\t',r, '\t',f1)
        
        
    def bc5cdr(self):

        def str2dic(answer):
            label = {}
            if '\n' in answer and ': ' in answer:
                tmp = answer.split('\n')
                # print('tmp',tmp)
                tmp = [i for i in tmp if i != '']
                # print('tmp', tmp)
                for ii in tmp:
                    # print("ii", ii)
                    if len(ii.split(': ')[0].split()) == 1:
                        # print('tmp', tmp)
                        # print('answer', answer)
                        label[ii.split(': ')[0]] = ii.split(': ')[-1].split('; ')
                return label
            else:
                return label
        # /data/qjw/ms-swift-main/test_result/NER/glm4-9b/v0-20241009-135227/NER-bc5cdr-test-glm4base_102400.json
        inp_pre_test = self.pred_path+'/NER-bc5cdr-test-glm4base_94400.json'
        oup_pre_test = self.temp_path+'/NER-bc5cdr-test_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-bc5cdr-test.jsonl'
        oup_tru_test = self.temp_path+'/NER-bc5cdr-test_dic_gold.jsonl'
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        
        self.com_prf(oup_tru_test, oup_pre_test)
    
    def biored(self):

        def str2dic(answer):
            label = {}
            if '\n' in answer and ': ' in answer:
                tmp = answer.split('\n')
                # print('tmp',tmp)
                tmp = [i for i in tmp if i != '']
                # print('tmp', tmp)
                for ii in tmp:
                    # print("ii", ii)
                    if len(ii.split(': ')[0].split()) == 1:
                        # print('tmp', tmp)
                        # print('answer', answer)
                        label[ii.split(': ')[0]] = ii.split(': ')[-1].split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-biored-test-glm4base_102400.json'
        oup_pre_test = self.temp_path+'/NER-biored-test_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-biored-test.jsonl'
        oup_tru_test = self.temp_path+'/NER-biored-test_dic_gold.jsonl'
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        
        self.com_prf(oup_tru_test, oup_pre_test)
    
    def chemner(self):
        def str2dic(answer):
            label = {}
            # print('answer',answer)
            if ': ' in answer:
                tmp = answer.split(': ')[-1]
                # print('tmp',tmp)
                label['Chemical'] = tmp.split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-chemdner-test-glm4base_94400.json'
        oup_pre_test = self.temp_path+'/NER-chemdner-test_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-chemdner-test.jsonl'
        oup_tru_test = self.temp_path+'/NER-chemdner-test_dic_gold.jsonl'
        
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        self.com_prf(oup_tru_test, oup_pre_test)
    
    def ncbidis(self):
        def str2dic(answer):
            label = {}
            # print('answer',answer)
            if ': ' in answer:
                tmp = answer.split(': ')[-1]
                # print('tmp',tmp)
                label['Chemical'] = tmp.split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-ncbi_disease-test-glm4base_94400.json'
        oup_pre_test = self.temp_path+'/NER-ncbi_disease-test_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-ncbi_disease-test.jsonl'
        oup_tru_test = self.temp_path+'/NER-ncbi_disease-test_dic_gold.jsonl'
        
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        self.com_prf(oup_tru_test, oup_pre_test)    
    
    def nlmgene(self):
        def str2dic(answer):
            label = {}
            # print('answer',answer)
            if ': ' in answer:
                tmp = answer.split(': ')[-1]
                # print('tmp',tmp)
                label['Chemical'] = tmp.split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-nlm_gene-test-glm4base_102400.jsonl'
        oup_pre_test = self.temp_path+'/NER-nlm_gene-test_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-nlm_gene-test.jsonl'
        oup_tru_test = self.temp_path+'/NER-nlm_gene-test_dic_gold.jsonl'
        
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        self.com_prf(oup_tru_test, oup_pre_test)   
    
    def tmvar(self):
        def str2dic(answer):
            label = {}
            # print('answer',answer)
            if ': ' in answer:
                tmp = answer.split(': ')[-1]
                # print('tmp',tmp)
                label['Chemical'] = tmp.split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-tmvar_v1-test-glm4base_102400.jsonl'
        oup_pre_test = self.temp_path+'/NER-tmvar_v1-test_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-tmvar_v1-test.jsonl'
        oup_tru_test = self.temp_path+'/NER-tmvar_v1-test_dic_gold.jsonl'
        
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        self.com_prf(oup_tru_test, oup_pre_test) 
    
    def cmeee(self):
        def str2dic(answer):
            label = {}
            # print('answer',answer)
            if '\n' in answer and '：' in answer:
                tmp = answer.split('\n')
                # print('tmp',tmp)
                # exit()
                tmp = [i for i in tmp if i != '']
                # print('tmp', tmp)
                for ii in tmp:
                    # print("ii", ii)
                    # print("ii.split('：')[0].split()", ii.split('：')[0].split())
                    # exit()
                    if len(ii.split('：')[0].split()) == 1:
                        # print('tmp', tmp)
                        label[ii.split('：')[0]] = ii.split('：')[-1].split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-cmeee_v2-dev-glm4base_94400.json'
        oup_pre_test = self.temp_path+'/NER-cmeee_v2-dev_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-cmeee_v2-dev.jsonl'
        oup_tru_test = self.temp_path+'/NER-cmeee_v2-dev_dic_gold.jsonl'
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        
        self.com_prf(oup_tru_test, oup_pre_test)

    def imcs(self):
        def str2dic(answer):
            label = {}
            # print('answer',answer)
            if '\n' in answer and '：' in answer:
                tmp = answer.split('\n')
                # print('tmp',tmp)
                # exit()
                tmp = [i for i in tmp if i != '']
                # print('tmp', tmp)
                for ii in tmp:
                    # print("ii", ii)
                    # print("ii.split('：')[0].split()", ii.split('：')[0].split())
                    # exit()
                    if len(ii.split('：')[0].split()) == 1:
                        # print('tmp', tmp)
                        label[ii.split('：')[0]] = ii.split('：')[-1].split('; ')
                return label
            else:
                return label
        
        inp_pre_test = self.pred_path+'/NER-imcs_v2_ner-dev_answer.jsonl'
        oup_pre_test = self.temp_path+'/NER-imcs_v2_ner-dev_dic.jsonl'
        
        inp_tru_test = self.gold_path+'/NER-imcs_v2_ner-dev.jsonl'
        oup_tru_test = self.temp_path+'/NER-imcs_v2_ner-dev_dic_gold.jsonl'
        
        outfile = open(oup_pre_test, 'w', encoding='utf-8')
        with open(inp_pre_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["answer"]
                sentlabel = str2dic(answer)
                line["answer"] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        
        outfile = open(oup_tru_test, 'w', encoding='utf-8')
        with open(inp_tru_test, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)
                # print(type(line), line)
                answer = line["conversation"][0]['assistant']
                sentlabel = str2dic(answer)
                line["conversation"][0]['assistant'] = sentlabel
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
        outfile.close()
        
        self.com_prf(oup_tru_test, oup_pre_test)




gold_path = '/data/qjw/ms-swift-main/test_data/NER'
pred_path = '/data/qjw/ms-swift-main/5000other_testall'

ner_eval = NER_eval(gold_path,pred_path)
print('\nNER_bc5cdr')
ner_eval.bc5cdr()

# print('\nNER_biored')
# ner_eval.biored()

print('\nNER_cmeee')
ner_eval.cmeee()


print('\nNER_chemner')
ner_eval.chemner()

print('\nNER_ncbidis')
ner_eval.ncbidis()


