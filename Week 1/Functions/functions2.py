import array

numbers = array.array('i', [1, 2, 3])


def initArray(*args):
    myList = []
    for a in args:
        myList = args
    print(myList)

initArray(1, 2, 3)