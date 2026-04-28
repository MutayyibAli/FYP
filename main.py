import torch  # Import PyTorch library
import time  # Import time library for measuring execution time

from data_preprocessing.preprocessor import Preprocessor


def main():
    initial_setup()
    start = time.time()  # Record the start time of the program
    Preprocessor().run()  # Create an instance of the Preprocessor class and run it
    end = time.time()  # Record the end time of the program
    print(f"Preprocessing time: {end - start} seconds")


def initial_setup():
    # Enable TensorFloat-32 (TF32) for faster matrix operations on compatible NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Make the main function run when the script is executed
if __name__ == "__main__":
    main()
