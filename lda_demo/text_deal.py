"""
自然语言处理---文本预处理
"""
import jieba
import pandas as pd

"""
加载初始数据信息
str:文件传输路径
index:所需真实值索引列表
"""


def read_data(str, index):
    dream_data = pd.read_csv(str)
    return dream_data.values[:, index]


"""
去掉文本中的空格
input:our_data为list文本数据
output:去除空格后的文本list
"""


def remove_blank_space(contents):
    contents_new = map(lambda s: s.replace(' ', ''), contents)
    return list(contents_new)


"""
判断单词是否为中文
input:word单个单词
output:是中文True,不是中文False
"""


def is_chinese(word):
    if word >= u'\u4e00' and word <= u'\u9fa5':
        return True
    else:
        return False


"""
判断短句是否为纯中文
input:words短句
output:是中文True,不是中文False
"""


def is_chinese_words(words):
    for word in words:
        if word >= u'\u4e00' and word <= u'\u9fa5':
            continue
        else:
            return False
    return True


"""
将文本数据格式化去除非中文字符
input:contents list结构的文本数据
output:去除非中文字符的数据
"""


def format_contents(contents):
    contents_new = []
    for content in contents:
        content_str = ''
        for i in content:
            if is_chinese(i):
                content_str = content_str + i
        contents_new.append(content_str)
    return contents_new


"""
对文本进行jieba分词
input:contents文本list
output:分词后的文本list
"""


def cut_words(contents):
    cut_contents = map(lambda s: list(jieba.lcut(s)), contents)
    return list(cut_contents)


"""
去除停用词/标点符号
input:contents文本list(list中保存list)
output:去除停用词后的文本list
"""


def drop_stopwords(contents):
    # 初始化获取停用词表
    stop = open('./data/word_deal/stop_word_cn.txt', encoding='utf-8')
    stop_me = open('./data/word_deal/stop_one_mx.txt', encoding='utf-8')
    key_words = open('./data/word_deal/key_words.txt', encoding='utf-8')
    #分割停用词/自定义停用词/关键词
    stop_words = stop.read().split("\n")
    stop_me_words = stop_me.read().split("\n")
    key_words = key_words.read().split("\n")
    #定义返回后的结果
    contents_new = []
    #遍历处理数据
    for line in contents:
        line_clean = []
        for word in line:
            if (word in stop_words or word in stop_me_words) and word not in key_words:
                continue
            if is_chinese_words(word):
                line_clean.append(word)
        contents_new.append(line_clean)
    return contents_new


def deal_chinese_content():
    # 获取处理的文本list
    real_contents = read_data("./data/set/contents_real_dream.csv", 0)
    # 去除空格
    contents = remove_blank_space(real_contents)
    # 获取分词结果
    contents = cut_words(contents)
    # 去除标点符号
    # contents = format_contents(contents)
    # 去除停用词
    contents = drop_stopwords(contents)
    # 查看结果
    for r_content, content in zip(real_contents, contents):
        print("处理前：", r_content)
        print("处理后：", "/".join(content))
        print("----------------------------------------------------------------------")
