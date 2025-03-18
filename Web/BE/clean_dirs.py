import os
import shutil


if __name__ == "__main__":
    # Clean directories
    shutil.rmtree("processed")
    shutil.rmtree("json")
    os.mkdir("processed")
    os.mkdir("json")
    print("Cleaned directories.")
