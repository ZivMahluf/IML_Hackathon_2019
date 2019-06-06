from PreProcessor import PreProcessor


if __name__ == '__main__':
    processed_data, labels = PreProcessor.pre_process()
    processed_data.to_csv('processed/processed_data.csv')
    labels.to_csv('processed/processed_labels.csv')
    print('done!')
