import argparse

import numpy as np
import matplotlib.pyplot as plt


def main(ifile: str, ofile: str) -> None:
    with open(ifile, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.split(' ')[:-1], lines))
        lines = list(map(lambda x: list(map(float, x)), lines))
        data = np.array(lines, dtype=np.float32)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    img = ax.imshow(data, cmap="inferno")
    fig.colorbar(img, ax=ax)

    plt.savefig(ofile, dpi=300)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-i", "--input", type=str)
    argparser.add_argument("-o", "--output", type=str)

    args = argparser.parse_args()

    main(args.input, args.output)
