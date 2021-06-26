from ESN import ESN
import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))

def matrix_to_string(matrix):
    lines = ''
    rows = len(matrix)
    try:
        columns = len(matrix[0])
    except:
        for i in range(rows):
            lines += format(matrix[i], '.32f') + '\n'
        return lines
    for i in range(rows):
        row = ''
        for j in range(columns):
            row += format(matrix[i][j], '.32f')
            if j < columns-1:
                row += ' '
            else:
                row += '\n'
        lines += row
    return lines

def esn_to_string(ESN):
    return matrix_to_string(ESN.Win) + ':' + matrix_to_string(ESN.W) + ':' + matrix_to_string(ESN.Wout) + ':' + matrix_to_string(ESN.Wfb)

def save_esn(ESN, filepath):
    #try:
    f = open(BASEDIR + '/' + filepath, 'w')
    f.write(esn_to_string(ESN))
    f.close()
    #except:
        #print("Could not save network")

def load_esn(filepath):
    # Have constructor of ESN that takes the matrixes
    pass