from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import io
import tkinter.font as tkFont
import numpy as np
import pandas as pd
import random


def doc2vec_search(row_index):
    # destroy former widgets inside answerframe
    for widget in answer_frame.winfo_children():
        widget.destroy()

    function_font_style = tkFont.Font(family="微軟正黑體", size=11, weight=tkFont.BOLD)

    word_tokens = search_var.get().lower().split(" ")
    token_vector = model.infer_vector(word_tokens)
    sims = model.docvecs.most_similar([token_vector])
    similar_tweet_list = []
    for i, j in enumerate(sims):
        data_row = j[0]
        print(data_row)
        if i != 0:
            text = "By " + str(data.loc[data_row, 'retweet_from']) + ": " + str(data.loc[data_row, 'retweet_text']) + "\""
            similar_tweet_list.append(text)

    row_index += 2
    Label(answer_frame, text="The most similar tweet is....",
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')

    # the nytimes was thrilled by obama talking to communist dictator castro and horrified by trump talking to the elected
    # obama turned his back on taiwan
    for i, j in enumerate(similar_tweet_list):
        row_index += 1
        Label(answer_frame, text=str(i+1) + ". " + j + "\n",
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')


def similarity_search(row_index, last_row_count):
    function_font_style = tkFont.Font(family="微軟正黑體", size=10, weight=tkFont.BOLD)
    # destroy former widgets inside answerframe
    for widget in answer_frame.winfo_children():
        widget.destroy()

    tfidf_text_list = data.loc[:, 'stopwords_removed_retweet_text'].tolist()
    for i, j in enumerate(tfidf_text_list):
        if type(j) == np.nan:
            tfidf_text_list[i] = ""
        else:
            tfidf_text_list[i] = str(j).lower()

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(tfidf_text_list)
    random_data_row = random.randint(0, len(data))
    test_data = [tfidf_text_list[random_data_row]]
    test_data_vector = vectorizer.transform(test_data)
    similarity = cosine_similarity(test_data_vector, tfidf).flatten().tolist()
    sort_data = pd.DataFrame({
        'index': [a for a in range(0, len(similarity))],
        'similarity': [np.around(j, decimals=4) for i, j in enumerate(similarity)]
    })
    sort_data = sort_data.sort_values(by=['similarity'], ascending=False)
    similar_index = sort_data.iloc[1:6, 0].tolist()

    # original search function
    word_tokens = data.loc[random_data_row, 'retweet_text'].lower().split(" ")
    token_vector = model.infer_vector(word_tokens)
    sims = model.docvecs.most_similar([token_vector])
    similar_tweet_list = []
    for i, j in enumerate(sims):
        data_row = j[0]
        print(data_row)
        if i != 0:
            text = "By " + str(data.loc[data_row, 'retweet_from']) + ": " + str(
                data.loc[data_row, 'retweet_text']) + "\""
            similar_tweet_list.append(text)

    row_index += 1
    Label(answer_frame, text=str(data.loc[random_data_row, 'retweet_text'])[:120],
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')

    row_index += 1
    Label(answer_frame, text="=====================================",
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')

    row_index += 1
    Label(answer_frame, text="Doc2Vec:",
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')

    for i, j in enumerate(similar_tweet_list):
        row_index += 1
        j = j[:120]
        Label(answer_frame, text=str(i + 1) + ". " + j,
              font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')
        if i == 4:
            break

    row_index += 1
    Label(answer_frame, text="=====================================",
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')

    row_index += 1
    Label(answer_frame, text="TF-IDF:",
          font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')

    for i, j in enumerate(similar_index):
        row_index += 1
        Label(answer_frame, text=str(i + 1) + ". By " + str(data.loc[int(j), 'retweet_from']) + ": " + str(data.loc[int(j), 'retweet_text'])[:120],
              font=function_font_style).grid(row=row_index, column=0, columnspan=3, sticky='w')



if __name__ == '__main__':
    last_row_count = 0
    # gensim models and retweet data set up
    model = gensim.models.doc2vec.Doc2Vec.load('./gensim_models/doc2vec_model_retweet_new')
    data = pd.read_csv("./datasets/13_retweet_text_doc2vec_new.csv", sep='\t')
    # stopwords_removed_retweet_text
    tfidf_data = pd.read_csv("./datasets/15_retweet_text_tfidf_robert_version.csv", sep=',')
    data.loc[:, 'stopwords_removed_retweet_text'] = tfidf_data.loc[:, 'stopwords_removed_retweet_text']

    # Basic Settings
    win = Tk()
    title_font_style = tkFont.Font(family="微軟正黑體", size=12)
    normal_font_style = tkFont.Font(family="微軟正黑體", size=10)
    note_font_style = tkFont.Font(family="微軟正黑體", size=8)
    blank_font_style = tkFont.Font(size=3)
    license_font_style = tkFont.Font(family="微軟正黑體", size=6)
    win.title("Similar Tweet Search (Trump Edition)")
    win.geometry("800x550+100+100")
    main_frame = Frame(win)
    main_frame.pack(side='top', fill=X, padx=20, pady=5)
    answer_frame = Frame(win)
    answer_frame.pack(side='top', fill=X, padx=20)
    row_index = 0

    # title section
    row_index += 1
    Label(main_frame, text="\n【Similar Tweet Search (Trump Edition)】\n ",
          font=title_font_style).grid(row=row_index, column=0, columnspan=3)

    search_var = StringVar()
    row_index += 1
    Label(main_frame,
          text="Please enter the tweet you would like to search：",
          font=note_font_style).grid(row=row_index, column=0, sticky='w')

    # search entry
    search_entry = Entry(main_frame, textvariable=search_var, font=title_font_style, width=60).grid(
        row=row_index,
        column=1,
        sticky='w')

    # execute button
    Button(main_frame, text="Let's go!!!!!", font=normal_font_style,
           bg='skyblue', width=10, height=1, command=lambda: doc2vec_search(row_index)).grid(row=row_index, column=3,
                                                                                               sticky='w')

    # space row
    row_index += 1
    Label(main_frame, text="                                                                                        "
                           "                                                                                                                             ",
          font=blank_font_style).grid(row=row_index, column=0, columnspan=2)

    # execute button: random search
    row_index += 1
    Button(main_frame, text="Random Search!!!!!", font=normal_font_style,
           bg='skyblue', width=20, height=1, command=lambda: similarity_search(row_index, last_row_count)).grid(row=row_index, column=3,
                                                                                             sticky='w')

    # space row
    row_index += 1
    Label(main_frame, text="                                                                                        "
                           "                                                                                                                             ",
          font=title_font_style).grid(row=row_index, column=0, columnspan=2)


    # activate the app
    win.mainloop()



