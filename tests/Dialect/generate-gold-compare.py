#!/usr/bin/env python3

import sys
import subprocess

passFlag = sys.argv[1]
print("Will run pass", passFlag, " on input file.")
inputFile = sys.argv[2]
print("Input file:", inputFile)
goldFile = inputFile + ".gold"
print("Will generate gold file:", goldFile)
print("")

# Run lapis-opt once to generate the gold file
print("Running lapis-opt...")
command = " ".join(["lapis-opt", passFlag, inputFile, ">", goldFile])
subprocess.run(command, shell = True)
print("The gold file:")
subprocess.run(["cat", goldFile])
print("Inserting RUN line into input file...")
runLine = "// RUN: %lapis-opt %s " + passFlag + " | diff %s.gold -"

f1 = open(inputFile, 'r')
inputContents = f1.read()
f1.close()
# If input file already contained a //RUN line, delete it
# before inserting the new one.
firstLine = inputContents[:inputContents.find('\n') + 1]
if firstLine.find('// RUN') != -1:
    inputContents = inputContents[len(firstLine):]
f2 = open(inputFile, 'w')
f2.write(runLine + "\n" + inputContents)
f2.close()

