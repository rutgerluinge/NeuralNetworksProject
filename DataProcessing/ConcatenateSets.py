import csv
import os

def main():
    path = os.getcwd() + '/Datasets/Correct/'
    files = os.listdir(path)
    files = [file_name for file_name in files if file_name.endswith('.csv')]
    
    csv_out = csv.writer(open(path + 'usage.csv', 'w', newline=''))
    for csv_file in files:
        file_in = open(path + csv_file, 'r', newline='')
        csv_in = csv.reader(file_in)
        for row in csv_in:
            csv_out.writerow(row)
        file_in.close()


if __name__ == '__main__':
    main()