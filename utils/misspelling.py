import pandas as pd
import pymorphy2
import fasttext
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm


def misspelling(commentData, vocabData):
    morph = pymorphy2.MorphAnalyzer()

    necessary_part = {"NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"}
    c_df = pd.read_csv(commentData, usecols=['stars', 'comment'], index_col=False) # Comments DataFrame
    c_df['comment'] = c_df['comment'].str.split(' ')
    v_df = pd.read_csv(vocabData, index_col=False).set_index('index') # Vocabulary DatFrame
    sentences = []

    # Normalization
    for line in v_df['Lemma']:
        sentences.append(text_to_word_sequence(line))

    for i in tqdm(range(len(sentences))):
        sentence = []
        for el in sentences[i]:
            p = morph.parse(el)[0]
            if p.tag.POS in necessary_part:
                sentence.append(p.normal_form)
        sentences[i] = sentence
    sentences = [x[0] for x in sentences if x]
    with open('data/temp_sentences.txt', 'w') as output_sentences:
        for word in sentences:
            output_sentences.write(f'{word}\n')

    # Training
    model = fasttext.train_unsupervised(input='data/temp_sentences.txt', minCount=1, model='skipgram', loss='hs')

    # Inference
    for line in c_df['comment']:
        for word in line:
            if word in model:
                print(word)