from data_processor import DataProcessor

def main():
    dp = DataProcessor()
    dp.process_train_test()

    for markers, video, audio in dp.process_to_model():
        print(len(markers), len(video), len(audio))

if __name__ == '__main__':
    main()