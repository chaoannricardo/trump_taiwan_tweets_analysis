# Import Packages
import matplotlib.pyplot as plt
import monpa
import numpy as np
import os
import pandas as pd

# Configure Monpa Long cut function
def Monpa_LongCut(long_sentence, split_char):
    seg = []
    for item in long_sentence.split(split_char):
        if item != "\n": seg.extend(monpa.cut(str(item+split_char)))
    return seg[:-1]

# Change working directory
working_directory = '/home/user/GithubRepository/College/Courses/10801_MachineLearningAndDeepLearning/Assignments/Assignment_1/Assignment1_Pycharm/Submission/AdditionalData'
os.chdir(working_directory)

# import Chinese text dataset
pd.options.display.max_colwidth = 100000000
ChineseDataset = pd.read_csv('./ChineseDataset_Assignment1.txt', sep='^', header=None, dtype='str')
print(ChineseDataset.head(3))
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

# Create Dataframe with Word Counts
word_list_df = pd.DataFrame({'Word': [],
                             'TF_Count': [],
                             'DF_Count': []})

# Initialize value of overall_length_of_document
overall_length_of_documents = 0

# Tokenization with Monpa
# for i in range(len(ChineseDataset)):
for i in range(len(ChineseDataset)):
    print('== 第', str(i), '篇斷詞，TF-IDF 計算開始 ==')
    line = str(ChineseDataset.iloc[i, :])
    cut_list = Monpa_LongCut(line, '，')
    # Strip the blanks and \n
    result_list = []
    for index, item in enumerate(cut_list):
        item = item.strip()
        result_list.append(item)
    # Compute length of document, overall length of document
    length_of_document = len(result_list)
    overall_length_of_documents = overall_length_of_documents + length_of_document
    # Remove Stop words
    remain_words = list(filter(lambda a: a not in stop_words and a != '\n', result_list))
    # Append data into wordcount dataframe
    # Calculate DF(phase 1), if word is inside word_list, DF + 1
    unique_remain_words = list(set(remain_words))
    for index, word in enumerate(unique_remain_words):
        condition = (word_list_df['Word'] == word)
        if word in word_list_df.loc[:, 'Word'].tolist():
            word_list_df.loc[condition, 'DF_Count'] = word_list_df.loc[condition, 'DF_Count'] + 1
    # Calculate TF, DF(phase 2)
    for index, word in enumerate(remain_words):
        condition = (word_list_df['Word'] == word)
        # If word is not inside word list, TF = 1, DF = 1
        if word not in word_list_df.loc[:, 'Word'].tolist():
            word_list_df = word_list_df.append({'Word': word,
                                                'TF_Count': int(1),
                                                'DF_Count': int(1)}, ignore_index=True)
        # If word is inside word_list, TF = TF + 1
        else:
            word_list_df.loc[condition, 'TF_Count'] = word_list_df.loc[condition, 'TF_Count'] + 1

# Calculate TF / length of all document
word_list_df.loc[:, 'TF/len_all'] = word_list_df.loc[:, 'TF_Count'] / overall_length_of_documents
# Calculate IDF
word_list_df.loc[:, 'IDF'] = np.log(len(ChineseDataset) / (word_list_df.loc[:, 'DF_Count'] + 1))
# Calculate TF-IDF
# TF-IDF Calculation: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
word_list_df.loc[:, 'G_TF-IDF'] = word_list_df.loc[:, 'TF_Count'] * word_list_df.loc[:, 'IDF']
# Compute Rank
# pandas.DataFrame.rank: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html
word_list_df['Rank'] = word_list_df['TF_Count'].rank(method='first', ascending=False)
# Add overall length for double check usage
word_list_df['overall_length'] = overall_length_of_documents
# Sort dataframe to graph the plot
word_list_df = word_list_df.sort_values(by='Rank', ascending=True)

print('Overall length of all document is ', overall_length_of_documents)
print('Length of word dict is ', len(word_list_df))

# Graphing Process
plt.rcParams.update({'font.size':18})
plt.figure(figsize=[15, 10])
plt.plot(word_list_df.loc[:, 'Rank'], word_list_df.loc[:, 'TF_Count'], linewidth=5, color='orange')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Zipf Distribution Plot')
plt.savefig('./Zipf_Distribution_Plot.png')

# Output to csv file
word_list_df.to_csv('./ChineseDataset_TF_IDF.txt', header=True, index=False)

# With only word rank
word_list_df['TFIDF_Rank'] = word_list_df['G_TF-IDF'].rank(method='first', ascending=False)
# Sort dataframe to output G_TF-IDF Rank
word_list_df = word_list_df.sort_values(by='TFIDF_Rank', ascending=True)
word_list_df.loc[:, 'G_TF-IDF']= np.around(word_list_df.loc[:, 'G_TF-IDF'], decimals=4)
singlefile = word_list_df[['Word', 'G_TF-IDF']]
singlefile.to_csv('./ChineseDataset_word_rank.csv', header=True, index=False)
