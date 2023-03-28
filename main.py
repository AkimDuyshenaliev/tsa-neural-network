from dataCleaning import csvCleaning
from model import text_sentiment_neural_network

class FileNames:
    dataPath = 'data/stars_and_comments.csv'
    cleanDataPath = 'data/clean_data.csv'
    cleanTrainPath = 'data/clean_train_data.csv'
    cleanTestPath = 'data/clean_test_data.csv'


if __name__ == '__main__':
    text_sentiment_neural_network(
        testData=FileNames.cleanTestPath, 
        trainData=FileNames.cleanTrainPath)