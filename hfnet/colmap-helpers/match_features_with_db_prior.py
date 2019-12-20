import argparse
import os
from tqdm import tqdm

from internal import db_matching_images
from internal import frame_matching


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--min_num_matches', type=int, default=15)
    parser.add_argument('--num_points_per_frame', type=int, default=2500)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--npz_dir', required=True)

    # This argument lets us only look at the matches from a certain folder.
    # We want to avoid adding matches from other folders, e.g. query. This
    # filters images according to the prefix as stored in the db file.
    parser.add_argument('--image_prefix', required=True)

    parser.add_argument('--use_ratio_test', action='store_true')
    parser.add_argument('--ratio_test_values', type=str, default='0.85')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ratio_test_values = [float(v) for v in args.ratio_test_values.split(',')]
    print('Ratio test values to use:', ratio_test_values)
    outfiles = [open('matches{}.txt'.format(x), 'w+')
                for x in [int(i * 100) for i in ratio_test_values]]

    print('Looking for matching image pairs...')
    matching_image_pairs = db_matching_images.get_matching_images(
        args.database_file, args.min_num_matches, args.image_prefix)
    print('Got', len(matching_image_pairs), 'matching image pairs.')

    num_missing_images = 0
    for (name1, name2) in tqdm(matching_image_pairs, unit='pairs'):
        # Get npz instead of image files.
        npz1 = os.path.join(args.npz_dir, os.path.splitext(name1)[0] + '.npz')
        npz2 = os.path.join(args.npz_dir, os.path.splitext(name2)[0] + '.npz')
        image1 = os.path.join(args.image_dir, name1)
        image2 = os.path.join(args.image_dir, name2)

        # Some images might be missing, e.g. in the Robotcar case.
        if not os.path.isfile(image1) or not os.path.isfile(image2):
            num_missing_images += 1
            continue

        assert os.path.isfile(npz1), npz1
        assert os.path.isfile(npz2), npz2

        num_points = args.num_points_per_frame
        matches_for_different_ratios = frame_matching.match_frames(
            npz1, npz2, image1, image2, num_points,
            args.use_ratio_test, ratio_test_values, args.debug)

        if(args.use_ratio_test):
            assert len(matches_for_different_ratios) == len(ratio_test_values)

        for i, keypoint_matches in enumerate(matches_for_different_ratios):
            if len(keypoint_matches) > args.min_num_matches:
                outfiles[i].write(name1 + ' ' + name2 + '\n')
                for (match1, match2) in keypoint_matches:
                    outfiles[i].write(str(match1) + ' ' + str(match2) + '\n')
                outfiles[i].write('\n')

    for outfile in outfiles:
        outfile.close()

    print('Missing', num_missing_images, 'images skipped.')


if __name__ == '__main__':
    main()
