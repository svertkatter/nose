import sys
def main():
    print("Hello from inside Docker!")
    print("Python version:", sys.version.split()[0])

if __name__ == "__main__":
    main()