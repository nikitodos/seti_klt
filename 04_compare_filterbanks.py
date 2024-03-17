import numpy as np
import sys
import argparse

from your.formats.filwriter import make_sigproc_object
from your import Your, Writer

def compare_matrices(f1, f2, t):
    input_data1 = Your(f1)
    input_data2 = Your(f2)
    
    header1=input_data1.your_header
    header2=input_data2.your_header
    print(header1, header2)
    
    if (header1.nbits != header2.nbits):
        print("Headers differ in nbits: ", header1.nbits, header2.nbits)
        raise Exception("headers nbits diff\n")
    if (header1.nspectra != header2.nspectra):
        print("Headers differ in nspectra: ", header1.nspectra, header2.nspectra)
        raise Exception("headers nspectra diff\n")

    matrix1=input_data1.get_data(nstart=0, nsamp=header1.nspectra)
    matrix2=input_data2.get_data(nstart=0, nsamp=header2.nspectra)

    print(matrix1.shape, matrix2.shape, matrix1.T.shape)

    if (t is True):
        print("numpy all close: ", np.allclose(matrix1.T, matrix2))
        print("numpy array equal: ", np.array_equal(matrix1.T, matrix2))
    else:
        print("numpy all close: ", np.allclose(matrix1, matrix2))
        print("numpy array equal: ", np.array_equal(matrix1, matrix2))
        
def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description = "Compares two filterbanks if and warns for differences"
        )
    parser.add_argument('-f1',
                        '--input_file',
                        action = "store" ,
                        help = "SIGPROC .fil input file",
                        required = True)
    parser.add_argument('-f2',
                        '--output_file',
                        action = "store" ,
                        help = "SIGPROC .fil processed output file",
                        required = True
                        )
    parser.add_argument('-t',
                        '--transposed',
                        action = "store_true" ,
                        help = "transposes input matrix for comparison"
                        )
    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    input_file1 = args.input_file
    input_file2 = args.output_file
    transposed  = args.transposed

    compare_matrices(f1 = input_file1, f2 = input_file2, t = transposed)