import sys
import os
import getopt


def data(fileName, fileOut,pos):
    f = open(fileName)
    w = open(fileOut, 'w')
    flag=False
    w.write("@relation DNA")
    w.write('\n')
    w.write("@attribute D string")
    w.write('\n')
    w.write("@attribute class	  {1,-1}")
    w.write('\n')
    w.write("@data")
    w.write('\n')
    doc=f.readlines()
    pos=int(pos)

    for line in doc:
        line = line.upper().strip()
        if line.startswith(">"):
            continue
        elif pos >0:
            w.write(line+",1")
            w.write("\n")
            pos-=1
        else:
            w.write(line+",-1")
            w.write("\n")

    f.close()
    w.close()


def main(argv):
    opts, args = getopt.getopt(sys.argv[1:], "hf:a:l:")
    input_file = ''
    output_file = ''
    pos=0

    for op, value in opts:
        if op == "-f":
            input_file = value
        elif op == "-a":
            output_file = value
        elif op == "-l":
            pos = value
        else:
            sys.exit()

    data(input_file, output_file,pos)


if __name__ == '__main__':
    main(sys.argv)










