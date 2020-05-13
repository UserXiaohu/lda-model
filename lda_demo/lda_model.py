from lda_demo import text_deal as td
from gensim import corpora, models
import gensim

"""
读取text文件
intput:url
output:list结构的文本数据
"""


def read_text_file(url):
    dream_text = open(url, 'r+', encoding='utf-8')
    return dream_text.read().split("\n\n")


"""
停用词/分词
"""


def deal_words(contents):
    # 去除空格
    contents = td.remove_blank_space(contents)
    # 获取分词结果
    contents = td.cut_words(contents)
    # 去除停用词
    contents = td.drop_stopwords(contents)
    return contents


"""
LDA模型
from gensim import corpora, models
import gensim
"""


def lad_model(train_data):
    # 读取预测分类数据
    test_data = deal_words(read_text_file('./data/set/test_text.txt'))
    # 拼接提取词典库
    contents = train_data + test_data
    # 根据文本获取词典
    dictionary = corpora.Dictionary(contents)

    # 词典创建语料库
    corpus = [dictionary.doc2bow(doc) for doc in train_data]

    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=30, id2word=dictionary,
                                          passes=2)  # 调用LDA模型，请求潜在主题数30；训练语料库2次

    data = lda.print_topics(num_topics=3, num_words=5)

    for item in data:
        print(item)  # 打印主题，10个主题，20个单词
        print("--------------------split line---------------------")

    # 测试主题分类
    test_vec = [dictionary.doc2bow(doc) for doc in test_data]
    target = {}
    for i, item in enumerate(test_vec):
        topic = lda.get_document_topics(item)
        keys = target.keys()
        print('第',i+1,'条记录分类结果:',topic)

if __name__ == '__main__':
    # 帖吧爬虫数据集处理
    contents = read_text_file('./data/set/train_text.txt')
    # 文本处理
    contents = deal_words(contents)
    # LDAmodel
    lad_model(contents)
