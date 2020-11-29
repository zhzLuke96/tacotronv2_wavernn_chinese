import os
import argparse
from wavernn_gen import wavernn
from tacotron_synthesize import tacotron

cwd = os.getcwd()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='', help='text to synthesis.')
    args = parser.parse_args()
    text = args.text

    checkpoint_path, idx, time = tacotron(text)

    melPth = os.path.abspath('./tacotron_inference_output/step-206500-' + idx +
                             '-mel-pred.npy')
    wavernn(melPth)
