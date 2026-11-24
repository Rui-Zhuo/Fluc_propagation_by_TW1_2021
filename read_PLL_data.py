"""Convert one explicitly selected PLL MAT observation into text."""
from tw1.io import convert_pll


def main(date='s1a26x', station='ur', channel='chan3', output=None):
    return convert_pll(date, station, channel=channel, output=output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='s1a26x')
    parser.add_argument('--station', default='ur')
    parser.add_argument('--channel', default='chan3')
    parser.add_argument('--output')
    main(**vars(parser.parse_args()))
