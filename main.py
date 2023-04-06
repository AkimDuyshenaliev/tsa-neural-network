from dataCleaning import csvCleaning, freqVocab, applyVocab
from utils.trainFT import trainFastText
# from model import text_sentiment_neural_network

class FileNames:
    dataPath = 'data/data-to-clean/stars_and_comments.csv'
    freqrnc = 'data/data-to-clean/freqrnc2011.csv'
    cleanDataPath = 'data/clean_data.csv'
    cleanTrainPath = 'data/clean_train_data.csv'
    cleanTestPath = 'data/clean_test_data.csv'
    cleanVocab = 'data/freqVocabClean.csv'
    ftRuModel = 'data/cc.ru.300.bin'


if __name__ == '__main__':
    print('1 - cvsCleaning\n2 - freqVocab\n3 - applyVocab\n4 - Train FastText')
    choice = int(input('Choose what to do: '))
    match choice:
        case 1:
            csvCleaning(FileNames.dataPath)
        case 2:
            freqVocab(FileNames.freqrnc)
        case 3:
            applyVocab(commentData=FileNames.cleanDataPath, vocabData=FileNames.cleanVocab)
        case 4:
            trainFastText(commentData=FileNames.cleanDataPath, vocabData=FileNames.cleanVocab)
        case _:
            print('No such thing')

    # text_sentiment_neural_network(
    #     testData=FileNames.cleanTestPath, 
    #     trainData=FileNames.cleanTrainPath)