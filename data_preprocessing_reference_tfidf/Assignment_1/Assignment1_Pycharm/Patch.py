import pandas as pd
import numpy as np
import os
import monpa
import matplotlib.pyplot as plt


# Configure Monpa Feature
def Monpa_LongCut(long_sentence, split_char):
    seg = []
    for item in long_sentence.split(split_char):
        if item != "\n": seg.extend(monpa.cut(str(item+split_char)))
    return seg[:-1]

# Change working directory
working_directory = 'C:/Users/ricardo/Documents/GitHub/College/Courses/10801_MachineLearningAndDeepLearning/Assignments/Assignment_1'
os.chdir(working_directory)

# import text document
pd.options.display.max_colwidth = 100000000
ChineseDataset = pd.read_csv('./ChineseDataset_Assignment1.txt', sep='^', header=None, dtype='str')
print(ChineseDataset.head(3))
print()
print('How many lines inside the document: ', len(ChineseDataset))

'''
Chinese stop word dictionary Reference: https://github.com/tomlinNTUB/Python-in-5-days
'''
# Create Chinese Stop Words Dictionary
stop_words = []
with open('./ChineseStopWord.txt', 'r', encoding='UTF-8') as file:
    for line in file.readlines():
        line = line.strip()
        stop_words.append(line)

word_list_df  = pd.read_csv('./ChineseDataset_TF_IDF.txt')


# Create Dataframe with Word Counts
overall_length_of_documents = word_list_df.loc[:,'overall_length']
word_list_df = word_list_df.drop(['TF/len_all','IDF','TF-IDF','Rank','overall_length'], axis=1)
word_list_df.loc[:,'DF_Count'] = 0
print(word_list_df.head())

# Tokenization with Monpa
# for i in range(len(ChineseDataset)):
for i in range(len(ChineseDataset)):
    print('== 【補丁】第', str(i), '篇斷詞，IDF 計算開始 ==')
    line = str(ChineseDataset.iloc[i,:])
    cut_list = Monpa_LongCut(line, '，')
    # Strip the blanks and \n
    result_list = []
    for index,item in enumerate(cut_list):
        item = item.strip()
        result_list.append(item)
    # Remove Stop words
    remain_words = list(filter(lambda a: a not in stop_words and a != '\n', result_list))
    # Append data into wordcount dataframe
    # Calculate DF(phase 1), if word is inside word_list, DF + 1, break
    unique_remain_words = list(set(remain_words))
    for index,word in enumerate(unique_remain_words):
        condition = (word_list_df['Word'] == word)
        word_list_df.loc[condition, 'DF_Count'] = word_list_df.loc[condition, 'DF_Count'] + 1

# Calculate TF / length of all document
word_list_df.loc[:,'TF/len_all'] = word_list_df.loc[:,'TF_Count'] / overall_length_of_documents
# Calculate IDF
word_list_df.loc[:,'IDF'] =  np.log(len(ChineseDataset) / (word_list_df.loc[:,'DF_Count'] + 1))
# Calculate TF-IDF
# TF-IDF Calculation: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
word_list_df.loc[:,'TF-IDF'] =  word_list_df.loc[:,'TF_Count']  * word_list_df.loc[:,'IDF']
# Compte Rank
# pandas.DataFrame.rank: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html
word_list_df['Rank'] = word_list_df['TF_Count'].rank(method='first', ascending=False)
# Add overall length for double check usage
word_list_df['overall_length'] = overall_length_of_documents
# Sort dataframe to graph the plot
word_list_df = word_list_df.sort_values(by='Rank', ascending=True)


print('Overall length of all document is ', overall_length_of_documents)
print('Length of word dict is ', len(word_list_df))

word_list_df.to_csv('./New_ChineseDataset_TF_IDF.csv', header=True, index=False)