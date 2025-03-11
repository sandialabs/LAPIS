import sys

def myprint(c):
    print(c, end='')

myprint('"')
while True:
    c = sys.stdin.read(1)
    if c == '':
        # EOF
        break
    if c == '"':
        myprint("\\\"")
    elif c == '\'':
        myprint("\\\'")
    elif c == '\n':
        myprint("\\n\"\n\"")
    elif c == '\t':
        myprint("    ")
    elif c == '\\':
        myprint("\\\\")
    else:
        myprint(c)
myprint('"')
print("")
