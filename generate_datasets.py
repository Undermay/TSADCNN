import argparse
import os
import numpy as np
from utils.data_loader import generate_motion_dataset


def save_dataset(segments, labels, path, segments_per_mode=None, segments_per_target=None, variant='clean'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, {
        'segments': segments,
        'labels': labels,
        'meta': {
            'num_segments': len(segments),
            'segments_per_mode': segments_per_mode,
            'segments_per_target': segments_per_target,
            'variant': variant
        }
    })
    print(f"Saved {len(segments)} segments to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate TSADCNN paper-style motion datasets")
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension (4 or 6)')
    parser.add_argument('--train_segments_per_mode', type=int, default=10000, help='Train segments per motion mode')
    parser.add_argument('--test_segments_per_mode', type=int, default=1000, help='Test segments per motion mode')
    parser.add_argument('--segments_per_target', type=int, default=5, help='Segments per target for contrastive positives')
    parser.add_argument('--generate_noisy', action='store_true', help='Also generate noisy datasets for robustness tests')
    parser.add_argument('--noisy_pos_std', type=float, default=1.0, help='Position noise std for noisy dataset')
    parser.add_argument('--noisy_vel_std', type=float, default=0.5, help='Velocity noise std for noisy dataset')
    args = parser.parse_args()

    # Generate training dataset (clean)
    train_segments, train_labels = generate_motion_dataset(
        sequence_length=args.sequence_length,
        input_dim=args.input_dim,
        segments_per_mode=args.train_segments_per_mode,
        segments_per_target=args.segments_per_target
    )

    # Generate testing dataset (clean)
    test_segments, test_labels = generate_motion_dataset(
        sequence_length=args.sequence_length,
        input_dim=args.input_dim,
        segments_per_mode=args.test_segments_per_mode,
        segments_per_target=args.segments_per_target
    )

    # Save clean
    train_path = os.path.join(args.output_dir, 'train_data.npy')
    test_path = os.path.join(args.output_dir, 'test_data.npy')
    save_dataset(train_segments, train_labels, train_path,
                 segments_per_mode=args.train_segments_per_mode,
                 segments_per_target=args.segments_per_target,
                 variant='clean')
    save_dataset(test_segments, test_labels, test_path,
                 segments_per_mode=args.test_segments_per_mode,
                 segments_per_target=args.segments_per_target,
                 variant='clean')

    # Optionally generate noisy datasets
    if args.generate_noisy:
        train_segments_noisy, train_labels_noisy = generate_motion_dataset(
            sequence_length=args.sequence_length,
            input_dim=args.input_dim,
            segments_per_mode=args.train_segments_per_mode,
            segments_per_target=args.segments_per_target,
            noise_std_pos=args.noisy_pos_std,
            noise_std_vel=args.noisy_vel_std
        )
        test_segments_noisy, test_labels_noisy = generate_motion_dataset(
            sequence_length=args.sequence_length,
            input_dim=args.input_dim,
            segments_per_mode=args.test_segments_per_mode,
            segments_per_target=args.segments_per_target,
            noise_std_pos=args.noisy_pos_std,
            noise_std_vel=args.noisy_vel_std
        )

        train_path_noisy = os.path.join(args.output_dir, 'train_data_noisy.npy')
        test_path_noisy = os.path.join(args.output_dir, 'test_data_noisy.npy')
        save_dataset(train_segments_noisy, train_labels_noisy, train_path_noisy,
                     segments_per_mode=args.train_segments_per_mode,
                     segments_per_target=args.segments_per_target,
                     variant='noisy')
        save_dataset(test_segments_noisy, test_labels_noisy, test_path_noisy,
                     segments_per_mode=args.test_segments_per_mode,
                     segments_per_target=args.segments_per_target,
                     variant='noisy')

    # Summary
    print("Generation complete.")
    print(f"Train(clean): {len(train_segments)} segments (per mode: {args.train_segments_per_mode}, total: {args.train_segments_per_mode * 5})")
    print(f"Test(clean):  {len(test_segments)} segments (per mode: {args.test_segments_per_mode}, total: {args.test_segments_per_mode * 5})")
    if args.generate_noisy:
        print(f"Train(noisy): {len(train_segments_noisy)} segments (per mode: {args.train_segments_per_mode}, total: {args.train_segments_per_mode * 5}, pos_std={args.noisy_pos_std}, vel_std={args.noisy_vel_std})")
        print(f"Test(noisy):  {len(test_segments_noisy)} segments (per mode: {args.test_segments_per_mode}, total: {args.test_segments_per_mode * 5}, pos_std={args.noisy_pos_std}, vel_std={args.noisy_vel_std})")


if __name__ == '__main__':
    main()