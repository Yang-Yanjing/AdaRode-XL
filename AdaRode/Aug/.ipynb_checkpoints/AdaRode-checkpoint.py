from Mutation.xss_attack import XssFuzzer
from Mutation.sql_attack import SqlFuzzer
from XLnet_Adapter import *
from Tools.Resultsaver import Results
import pandas as pd
import os
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
from collections import Counter
import json
from multiprocessing import Process, JoinableQueue, Queue, Lock
import multiprocessing as mp
import traceback
from torch.multiprocessing import Process, Queue, set_start_method
import math


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class MHM:
    def __init__(self, config, testset, labelset, resultsaver):
        self.config = config
        self.testset = testset
        self.labelset = labelset
        self.maxiter = config['parameters']['max_iterations']
        self.patience = config['parameters']['patience']
        self.accept = config['parameters']['accept_rate']
        self.resultsaver = resultsaver
        self.xsscount = 0
        self.sqlcount = 0
        self.xss_succ = 0
        self.sql_succ = 0
        self.xss_iter = []
        self.sql_iter = []
        self.succ_iter = []
        self.evasion_count = 0
        self.evasion_succ = 0
        self.attack_count = 0
        self.sql_choice = []
        self.xss_choice = []   
        self.sql_selection = [] 
        self.xss_selection = [] 
          
    def _filter_attack_order(self, data:str):
        # TODO：写完整的筛选算法
        flag = "selected"
        return flag
    
    def jud_type(self, vicstr:str, iteration:int) -> str:
        try:
            return self.typeset[iteration]
        except:
            # 常见的SQL关键字
            sql_keywords = [
                r"(?i)\bselect\b",
                r"(?i)\bfrom\b",
                r"(?i)\bwhere\b",
                r"(?i)\binsert\b",
                r"(?i)\binto\b",
                r"(?i)\bvalues\b",
                r"(?i)\bupdate\b",
                r"(?i)\bset\b",
                r"(?i)\bdelete\b",
                r"(?i)\bcreate\b",
                r"(?i)\balter\b",
                r"(?i)\bdrop\b",
                r"(?i)\bjoin\b",
                r"(?i)\binner\b",
                r"(?i)\bleft\b",
                r"(?i)\bright\b",
                r"(?i)\bouter\b",
                r"(?i)\bgroup\b",
                r"(?i)\bby\b",
                r"(?i)\border\b",
                r"(?i)\bhaving\b",
                r"(?i)\bunion\b",
                r"(?i)\bexec\b"
            ]

            # 常见的XSS特征
            xss_patterns = [
                r"(?i)<script.*?>.*?</script.*?>",
                r"(?i)<.*?javascript:.*?>",
                r"(?i)<.*?on\w+=.*?>",
                r"(?i)alert\(",
                r"(?i)document\.cookie",
                r"(?i)<iframe.*?>.*?</iframe.*?>",
                r"(?i)<img.*?src=.*?>",
                r"(?i)<.*?>",
                r"(?i)&lt;.*?&gt;"
            ]

            for pattern in sql_keywords:
                if re.search(pattern, vicstr):
                    return "sql"

            for pattern in xss_patterns:
                if re.search(pattern, vicstr):
                    return "xss"

            return "sql"
 
    def _formatResnote(self, _iter=None, _res=None, _prefix="  => "):
        status = _res['status'].lower()
        if status in ['s', 'r', 'a', 'f']:
            result_map = {
                's': 'SUCC!',
                'r': 'REJ.',
                'a': 'ACC!',
                'f': 'RB!'
            }
            return "%s iter %d, %s %s => %s (%d => %d, %.9f => %.9f) a=%.5f" % (
                _prefix, _iter, result_map[status], _res['old_object'], _res['new_object'],
                _res['old_pred'], _res['new_pred'],
                _res['old_prob'], _res['new_prob'], _res['alpha']
            )
        return None

    def getrecordMetrics(self):
        # Calculate individual metrics
        sql_asr = (self.sql_succ / self.sqlcount) * 100 if self.sqlcount > 0 else 0
        xss_asr = (self.xss_succ / self.xsscount) * 100 if self.xsscount > 0 else 0
        averageSQL = sum(self.sql_iter) / len(self.sql_iter) if len(self.sql_iter) > 0 else -1
        averageXSS = sum(self.xss_iter) / len(self.xss_iter) if len(self.xss_iter) > 0 else -1
        
        # Calculate overall metrics
        total_attacks = self.sqlcount + self.xsscount
        total_success = self.sql_succ + self.xss_succ
        overall_asr = (total_success / total_attacks) * 100 if total_attacks > 0 else 0
        overall_average_iter = (sum(self.sql_iter) + sum(self.xss_iter)) / (len(self.sql_iter) + len(self.xss_iter)) if (len(self.sql_iter) + len(self.xss_iter)) > 0 else -1
        
        log = "Immediately Record\n"
        log += "=" * 80 + "\n"
        log += "Current SQL attacks: {}, SQL ASR = {:.2f}%\n".format(self.sql_succ, sql_asr)
        log += "Current XSS attacks: {}, XSS ASR = {:.2f}%\n".format(self.xss_succ, xss_asr)
        log += "Current SQL num: {}\n".format(self.sqlcount)
        log += "Current XSS num: {}\n".format(self.xsscount)
        log += "Average SQL iterations: {}\n".format(averageSQL)
        log += "Average XSS iterations: {}\n".format(averageXSS)
        log += "Total attacks: {}\n".format(total_attacks)
        log += "Total successful attacks: {}\n".format(total_success)
        log += "Overall ASR: {:.2f}%\n".format(overall_asr)
        log += "Overall average iterations: {}\n".format(overall_average_iter)
        log += "=" * 80 + "\n"
        
        self.resultsaver.savelogDData(info=log)

    def count_elements(self, input_list):
        # 使用 Counter 统计元素出现的次数
        element_count = Counter(input_list)
        # 将 Counter 对象转换为字典
        return dict(element_count)

    def calculate_d_succ(self, samples):
        values = list(samples.values())
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)
        d_succ = std_dev / mean
        return d_succ
        
    def result_caculate(self):
        result = {
             # 添加一个总体结果字典
                "ASR": 0,
                "SAQ": 0,  # 平均成功攻击查询数
                "TAQ": 0,  # 平均查询步数
                "EVN": 0,  # 恶意逃逸计数
                "ER": 0,   # 恶意逃逸比率
                "count":0,
            "sql": {
                "AF": {}  # 填充SQL攻击类型的详细统计信息
            },
            "xss": {
                "AF": {}  # 填充XSS攻击类型的详细统计信息
            }
        }

        # Calculate overall metrics
        total_attacks = self.sqlcount + self.xsscount
        total_success = self.sql_succ + self.xss_succ
        overall_asr = (total_success / total_attacks) * 100 if total_attacks > 0 else 0
        result["ASR"] = overall_asr  # 填充总体ASR

        # 计算成功攻击的平均查询数（SAQ）
        success_average_queries = np.mean(self.succ_iter) if self.succ_iter else 0
        result["SAQ"] = success_average_queries  # 填充总体SAQ

        # 计算所有攻击的平均查询步数（TAQ）
        total_average_queries = (sum(self.sql_iter) + sum(self.xss_iter)) / (len(self.sql_iter) + len(self.xss_iter)) if (len(self.sql_iter) + len(self.xss_iter)) > 0 else -1
        result["TAQ"] = total_average_queries  # 填充总体TAQ

        # 恶意逃逸统计
        evasion_value = self.evasion_succ if self.evasion_succ > 0 else 0
        result["EVN"] = evasion_value  # 填充总体EVN

        # 计算恶意逃逸比率
        evasion_ratio = (self.evasion_succ / total_attacks) * 100 if total_attacks > 0 else 0
        result["ER"] = evasion_ratio  # 填充总体ER
        print(self.sql_choice)
        print(self.xss_choice)
        # 记录每种攻击类型的详细信息（AF）

        result["sql"]["AF"] = self.count_elements(self.sql_selection)  # 填充SQL攻击类型的详细统计信息
        result["xss"]["AF"] = self.count_elements(self.xss_selection)  # 填充XSS攻击类型的详细统计信息
        # print(result["sql"]["AF"])
        # 确保保存目录存在
        os.makedirs("Result", exist_ok=True)

        # 保存 SQL 攻击统计信息
        with open("Result/sql_attack_summary.json", "w", encoding="utf-8") as f_sql:
            json.dump(result["sql"]["AF"], f_sql, ensure_ascii=False, indent=4)

        # 保存 XSS 攻击统计信息
        with open("Result/xss_attack_summary.json", "w", encoding="utf-8") as f_xss:
            json.dump(result["xss"]["AF"], f_xss, ensure_ascii=False, indent=4)
        try:
            Dsql = self.calculate_d_succ(result["sql"]["AF"])
        except:
            Dsql = 0
        # print(result["xss"]["AF"])
        try:
            Dxss = self.calculate_d_succ(result["xss"]["AF"])
        except:
            Dxss = 0
        # Log the results
        log = "Results Summary\n"
        log += "=" * 80 + "\n"
        log += "Overall Attack Success Rate (ASR): {:.2f}%\n".format(overall_asr)
        log += "Average Attack Queries for Successful Attacks (SAQ): {:.2f}\n".format(success_average_queries)
        log += "Average Steps Required (TAQ): {:.2f}\n".format(total_average_queries)
        log += "Evasion Count (EVN): {}\n".format(evasion_value)
        log += "Evasion Ratio (ER): {:.2f}%\n".format(evasion_ratio)
        log += "Total Attack Count: {}\n".format(self.attack_count)
        log += "D_sql: {}\n".format(Dsql)
        log += "D_xss: {}\n".format(Dxss)
        log += "=" * 80 + "\n"
        result["count"] = self.attack_count
        self.resultsaver.savelogDData(info=log)

        return result  # 返回计算结果

    def _update_metrics_from_result(self, result, mcmc_result):
        """
        主进程更新攻击指标状态统计
        :param result: dict，包含 ori/label/type 等
        :param mcmc_result: mcmc() 返回结构
        """
        self.attack_count += 1

        evasion = result['label'] != 0
        data_type = result['type']

        if evasion:
            self.evasion_count += 1
            if mcmc_result['succ']:
                self.evasion_succ += 1

        if data_type == "sql":
            self.sqlcount += 1
            if mcmc_result['succ']:
                self.sql_succ += 1
            self.sql_iter.append(mcmc_result['used_iter'])
            self.sql_selection += mcmc_result['sql_choice']
        elif data_type == "xss":
            self.xsscount += 1
            if mcmc_result['succ']:
                self.xss_succ += 1
            self.xss_iter.append(mcmc_result['used_iter'])
            self.xss_selection += mcmc_result['xss_choice']

        if mcmc_result['succ']:
            self.succ_iter.append(mcmc_result['used_iter'])
        for line in mcmc_result['log']:
            self.resultsaver.savelogDData(info=line)

    def _replaceWords(self, _vic, _viclabel, datatype="xss", _prob_threshold=0.95, _candi_mode="allin", _victim=None):
        _prob_threshold = self.accept
        victim = _victim if _victim else self.victim  # ✅ 新增行

        if self._filter_attack_order(_vic) == "selected":
            candi_tokens = [_vic]
            if _candi_mode == "allin":
                if datatype == "xss":
                    Attacker = XssFuzzer(_vic)
                    for num in range(21):
                        resflag = Attacker.fuzz(num)
                        if resflag == -1:
                            Attacker.reset()
                            continue
                        candi_tokens.append(Attacker.current())
                        Attacker.reset()
                elif datatype == "sql":
                    Attacker = SqlFuzzer(_vic)
                    for num in range(12):
                        Attacker.fuzz(num)
                        candi_tokens.append(Attacker.current())
                        Attacker.reset()
                else:
                    print("GG!")
                    return 0
                _candi_tokens = candi_tokens
                candi_tokens.append(_vic)
                probs = np.array([victim.get_prob(sample) for sample in _candi_tokens])
                preds = [victim.get_pred(sample)[0] for sample in _candi_tokens]

                for i in range(len(candi_tokens)):
                    if preds[i] != _viclabel:
                        records = {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[i],
                                "old_prob": probs[0][_viclabel], "new_prob": probs[i][_viclabel],
                                "old_pred": preds[0], "new_pred": preds[i],
                                "adv_x": candi_tokens[i]}
                        return records
                
                candi_idx = np.argmin(probs[1:,_viclabel]) + 1
                candi_idx = int(candi_idx)
                if datatype == "xss":
                    self.xss_choice.append(candi_idx)
                elif datatype == "sql":
                    self.sql_choice.append(candi_idx)
                alpha = (1 - probs[candi_idx][_viclabel] + 1e-10) / (1 - probs[0][_viclabel] + 1e-10)

                if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                    records = {"status": "r", "alpha": alpha, "tokens": candi_tokens[candi_idx],
                            "old_object": _candi_tokens[0], "new_object": _candi_tokens[candi_idx],
                            "old_prob": probs[0][_viclabel], "new_prob": probs[candi_idx][_viclabel],
                            "old_pred": preds[0], "new_pred": preds[candi_idx]}
                    return records
                else:
                    if _candi_tokens[0] == _candi_tokens[candi_idx]:
                        records = {"status": "f", "alpha": alpha, "tokens": candi_tokens[candi_idx],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[candi_idx],
                                "old_prob": probs[0][_viclabel], "new_prob": probs[candi_idx][_viclabel],
                                "old_pred": preds[0], "new_pred": preds[candi_idx]}
                        return records
                    else:
                        records = {"status": "a", "alpha": alpha, "tokens": candi_tokens[candi_idx],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[candi_idx],
                                "old_prob": probs[0][_viclabel], "new_prob": probs[candi_idx][_viclabel],
                                "old_pred": preds[0], "new_pred": preds[candi_idx]}
                    return records
        else:
            print("Not regular")
           
    def mcmc(self, vic:str, viclabel:int, datatype="xss", _prob_threshold=0.95, _max_iter=100, _victim=None):
        if len(vic) <= 0:
            return {
                'succ': False,
                'tokens': None,
                'raw_tokens': None,
                'log': [],
                'used_iter': 0,
                'status': 'empty',
                'sql_choice': [],
                'xss_choice': []
            }

        tokens = vic
        jumpup = 0
        log_lines = []

        for iteration in range(1, 1 + _max_iter):
            res = self._replaceWords(
                _vic=tokens,
                _viclabel=viclabel,
                datatype=datatype,
                _prob_threshold=_prob_threshold,
                _victim=_victim
            )

            log_line = self._formatResnote(_iter=iteration, _res=res, _prefix="  >> ")
            if log_line:
                log_lines.append(log_line)

            if res['status'].lower() == 'f':
                jumpup += 1
                if jumpup > self.patience:
                    break

            if res['status'].lower() in ['s', 'a']:
                jumpup = 0
                tokens = res['tokens']
                if res['status'].lower() == 's':
                    return {
                        'succ': True,
                        'tokens': tokens,
                        'raw_tokens': tokens,
                        'log': log_lines,
                        'used_iter': iteration,
                        'status': 's',
                        'sql_choice': self.sql_choice[:],
                        'xss_choice': self.xss_choice[:]
                    }

        return {
            'succ': False,
            'tokens': None,
            'raw_tokens': None,
            'log': log_lines,
            'used_iter': iteration,
            'status': 'fail',
            'sql_choice': self.sql_choice[:],
            'xss_choice': self.xss_choice[:]
        }
    


    def exec(self, num_workers=3):
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

        res = {"ori_raw": [], "ori_label": [], "adv_raw": [], "adv_label": []}
        result_queue = Queue()
        update_lock = Lock()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total = len(self.testset)
        indices = list(range(total))
        chunks = [indices[i::num_workers] for i in range(num_workers)]  # 均分任务

        # 启动 worker
        processes = []
        for i in range(num_workers):
            p = Process(target=worker_fn_slice, args=(
                self.config, chunks[i],
                self.testset, self.labelset,
                result_queue, self.mcmc,
                self.jud_type, device
            ))
            p.start()
            processes.append(p)

        completed = 0
        start_time = time.time()
        timeout = 3600  # 设置运行时间上限（秒）

        with tqdm(total=total, position=1) as pbar:
            while completed < total:
                # 检查是否超时
                if time.time() - start_time > timeout:
                    print("\n⏰ Timeout reached. Terminating workers early.")
                    for p in processes:
                        p.terminate()
                        p.join()
                    break

                result = result_queue.get()

                if "error" in result:
                    print("\n🔥 Error:")
                    print(result["message"])
                    print(result["traceback"])
                    for p in processes:
                        p.terminate()
                        p.join()
                    raise RuntimeError("Worker crashed.")

                _res = result["mcmc_result"]
                pid = result["pid"]
                ori = result["ori"]
                label = result["label"]
                adv = _res["raw_tokens"] if _res["succ"] else ori

                with update_lock:
                    self.resultsaver.savelogDData(info=f"\nProcess {pid} finished EXAMPLE {completed} ...")
                    self.resultsaver.savelogDData(info=f"EXAMPLE {completed} {'SUCCEEDED!' if _res['succ'] else 'FAILED...'}")

                    self._update_metrics_from_result(
                        result={"label": label, "type": result["type"]},
                        mcmc_result=_res
                    )

                    res["ori_raw"].append(ori)
                    res["ori_label"].append(label)
                    res["adv_raw"].append(adv)
                    res["adv_label"].append(label)

                    if completed % 10 == 0 and completed != 0:
                        self.getrecordMetrics()

                completed += 1
                pbar.update(1)

        for p in processes:
            p.join()

        self.result_caculate()
        return res




def worker_fn_slice(config, indices, testset, labelset, result_queue, mcmc_fn, jud_type_fn, device):
    import os, traceback, torch

    pid = os.getpid()
    print(f"[PID {pid}] started with {len(indices)} samples")

    try:
        victim = RodeXLAdapter(modelpath=config['paths']['model_checkpoint'])
        victim.model.to(device)
        victim.model.eval()
        dummy = torch.randint(0, 1000, (1, 16)).to(device)
        with torch.no_grad():
            _ = victim.model(dummy)
        torch.cuda.synchronize()
    except Exception as e:
        result_queue.put({
            "error": True,
            "message": f"[Worker {pid}] Failed to init: {e}",
            "traceback": traceback.format_exc()
        })
        return

    for idx in indices:
        try:
            sample = testset[idx]
            label = labelset[idx]
            data_type = jud_type_fn(sample, idx)

            mcmc_result = mcmc_fn(
                vic=sample,
                viclabel=label,
                datatype=data_type,
                _max_iter=config['parameters']['max_iterations'],
                _victim=victim
            )

            result_queue.put({
                "idx": idx,
                "ori": sample,
                "label": label,
                "type": data_type,
                "mcmc_result": mcmc_result,
                "pid": pid
            })

        except Exception as e:
            result_queue.put({
                "error": True,
                "message": f"[Worker {pid}] Error at index {idx}: {e}",
                "traceback": traceback.format_exc()
            })



if __name__ == "__main__":
    for i in range(1,11):
        mp.set_start_method("spawn", force=True)
        config = load_config("Config/adv_config_AdaRode.yaml")
        test_data = pd.read_csv("/root/autodl-fs/Ada_RQ3/@TSE_attack_AdaRode/Data/Test/test_set_split"+str(i)+".csv")
        print("Data has been loaded...")
        
        data_list = test_data['Text'].tolist()
        label_list = test_data["Label"].tolist()
    
        resultsaver = Results("Test"+str(i), "AdaRode", "AdaData")
    
    
        attacker = MHM(config, data_list, label_list, resultsaver)
        workers = 3
        advdata = attacker.exec(num_workers = workers)
        import pandas as pd
    
        # 假设 advdata == res，里面有 ori_raw, adv_raw, ori_label, adv_label
        data_rows = []
    
        for adv_text, label in zip(advdata["adv_raw"], advdata["adv_label"]):
            # 根据 Label 赋值 type
            if label == 0:
                sample_type = "normal"
            elif label == 1:
                sample_type = "sql"
            elif label == 2:
                sample_type = "xss"
            else:
                sample_type = "unknown"  # 如果有其他 label, 默认填 unknown（可以改）
    
            data_rows.append({"Text": adv_text, "Label": label, "type": sample_type})

    # 将结果存储到 CSV 文件
    # output_path = os.path.join("Result", dataset, "Augmented"+data_name)
    # df = pd.DataFrame(data_rows)
    # df.to_csv(output_path, index=False)

    # print(f"Adversarial samples saved to {output_path}")
        


