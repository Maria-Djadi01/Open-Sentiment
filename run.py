import argparse

from ui import dashboard


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--share', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    dashboard.launch_demo(args.share)