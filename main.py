from dataCleaning import csvCleaning


class FileNames:
    dataPath = 'data/stars_and_comments.csv'


if __name__ == '__main__':
    csvCleaning(data=FileNames.dataPath)