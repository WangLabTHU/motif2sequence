import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter


# 功能：将hdf5文件中的启动子序列抽取并存储
def get_seq(location):
    with pd.HDFStore(location) as store:
        df = store.get('clean/majdet')
        sequence = df['seq']
    return sequence


# 功能：将list格式的序列以fa格式进行输出
def out_seq(sequence_file, file_name):
    with open(file_name, 'w') as f:
        for i, item in enumerate(sequence_file):
            f.write('>' + str(i) + '\n')
            f.write(item + '\n')


# 功能：将fimo的输出结果存入motif_info这个dict中，其中dict的key是一个motif的名称，
#       dict中每个key对应的value是包含这个motif的序列信息；
def info2dict(save=True, execute=True):
    if not execute:
        motif_info = np.load('fimo_dict.npy', allow_pickle=True).item()
        return motif_info
    fimo_result = pd.read_table('./fimo_yeast_nature_sequence/fimo.tsv')  # 读入fimo数据
    motif_info = {}  # 创建dict
    # 将信息放入dict格式的motif_info中，其中关键词为motif名称，关键词对应的即为包含该motif的序列信息
    for ind in fimo_result.index:
        if not motif_info.__contains__(fimo_result['motif_alt_id'][ind]):  # 初始化，即该motif未出现时，新建一个key
            motif_info[fimo_result['motif_alt_id'][ind]] = fimo_result.loc[ind]
        elif type(motif_info[fimo_result['motif_alt_id'][ind]]) == pd.core.series.Series:  # 第二次执行，将Series变为Dataframe
            motif_info[fimo_result['motif_alt_id'][ind]] = pd.DataFrame([motif_info[fimo_result['motif_alt_id'][ind]],
                                                                         fimo_result.loc[ind]])
        else:  # 以后在每个关键词自身的Dataframe中添加新的信息
            motif_info[fimo_result['motif_alt_id'][ind]] = \
                motif_info[fimo_result['motif_alt_id'][ind]].append(fimo_result.loc[ind])
    if save:  # 使用numpy库进行存储
        np.save('fimo_dict.npy', motif_info)
    return motif_info


# 将每个motif的位置信息以dict格式给出，这里给出的位置信息是开始与终止位置的平均；每个motif的名称是dict中的一个key；
def get_location(motif_info):
    location_record = {}
    for key in motif_info:
        try:  # 去除key为nan的异常情况
            np.isnan(key)
        except Exception:
            motif_location = []  # 记录motif所在的位置；
            for index, row in motif_info[key].iterrows():
                motif_location.append((row['start']))
            location_record[key] = motif_location
    return location_record


# 输入：以dict形式记录的motif及与之对应的位置信息，motif信息，输出模式（熵最大或是熵最小)，以及输出的熵最大/最小的motif数量
# 该函数的目的是，找到分布位置熵最大/最小的motif，然后将这个Motif及其位置信息输出出来；
def cal_entropy(location_record, motif_info, mode = 'minimum', number = 15):
    entropy_list = {}
    for key in location_record:
        # 计算不同位置的概率
        result = Counter(location_record[key]) # 得到不同位置出现的频率
        result = result.values()  #取出dict中的数值
        frequency = np.array(list(result))/sum(result) #计算每个位置出现的频率
        key_entropy = entropy(frequency)  #根据频率计算熵
        entropy_list[key] = key_entropy  #记录熵
    # 对entropy_list 排序
    part_entropy_list = {}
    if mode == 'minimum':   #选择熵最小模式
        entropy_list = {k: v for k, v in sorted(entropy_list.items(), key=lambda item: item[1])}  #熵从小到大排序
    elif mode == 'maximum':  #选择熵最大模式
        entropy_list = {k: v for k, v in sorted(entropy_list.items(), key=lambda item: item[1], reverse=True)}
    else:  #其它情况，无法排序，则输出空结果
        return entropy_list, part_entropy_list
    # 取出entropy最小/最大的number个motif
    m = 0
    for i, key in enumerate(entropy_list):
        if len(location_record[key]) > 100:  #含有该motif的序列数量必须大于100
            part_entropy_list[key] = motif_info[key]
            m = m + 1
        if m > number:
            break
    return entropy_list, part_entropy_list


def merge(part_entropy_list,seq):
    # 讲序列拼接成一个dataframe
    i = 0
    for key in part_entropy_list:
        if i == 0:
            part_dataframe = part_entropy_list[key]
            i = i + 1
        else:
            part_dataframe = part_dataframe.append(part_entropy_list[key],ignore_index=True)
    # 添加DNA序列
    sequence_info = []  #从数据集中抽取序列信息
    for i in range(len(part_dataframe)):
        sequence_num = int(part_dataframe.iloc[i]['sequence_name'])
        sequence_info.append(seq[sequence_num])

    part_dataframe['sequence'] = sequence_info

    return part_dataframe


if __name__ == '__main__':
    # 将启动子序列抽取出来，存入yeast_seq.fa
    file_location = './store.h5'
    seq = get_seq(file_location)
    out_seq(seq, 'yeast_seq.fa')

    # 这里使用fimo进行motif识别，具体代码如下：
    # ./fimo -oc fimo_yeast_nature_sequence -norc motif.meme yeast_seq.fa

    # 将fimo的输出结果存入motif_info这个dict中，其中dict的key是一个motif的名称，
    # dict中每个key对应的value是包含这个motif的序列信息；
    # 由于计算时间过长，因此将中间结果存储，直接读入即可（即将execute置为负数）
    motif_info = info2dict(save=False, execute=False)

    # 将每个motif在序列中的位置分布画出，查看最极端位置分布的motif：即仅在一个位置分布/在所有位置平均分布；
    location_record = get_location(motif_info)

    # 计算熵，得到熵最大或最小的一定数量的motif列表
    entropy_list, part_entropy_list = cal_entropy(location_record, motif_info, mode = 'minimum', number = 15)

    # 制作数据集；
    part_dataframe = merge(part_entropy_list,seq)

    # 添加轮廓项：这一项用作generator输入，类似于pix2pix中的简笔画输入；这里指的是特定motif的序列信息；
    sequence_info = []
    for i in range(len(part_dataframe)):
        start = int(part_dataframe.iloc[i]['start'])
        end = int(part_dataframe.iloc[i]['stop'])
        ori_seq = part_dataframe.iloc[i]['sequence']
        outline_seq = len(ori_seq[:start-1]) * 'Z' + ori_seq[start-1:end] + len(ori_seq[end::]) * 'Z'
        sequence_info.append(outline_seq)
    part_dataframe['sequence_outline'] = sequence_info


    # 输出数据
    part_dataframe.to_csv('part_dataset.csv')

