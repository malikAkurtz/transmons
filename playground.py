from Expression import *

def main():
    expression = Scalar(4)**2 * Op("n")

    print(expression)


if __name__=="__main__":
    main()