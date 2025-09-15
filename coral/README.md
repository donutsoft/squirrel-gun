TensorFlow Training (Image Classification)

This folder contains a simple, flexible TensorFlow/Keras training script for image classification with optional TFLite export.

Quick start

- Dataset layout (option 1: explicit train/val dirs):
  - dataset_root/
    - train/
      - class_a/
      - class_b/
    - val/
      - class_a/
      - class_b/

- Dataset layout (option 2: single folder with split):
  - dataset_root/
    - class_a/
    - class_b/

Examples

- Train a small CNN from scratch with split:
  python coral/train_tf.py --dataset data/images --epochs 10 --name exp_cnn

- Train with MobileNetV2 backbone, freeze base, and augment:
  python coral/train_tf.py \
    --dataset data/images \
    --base mobilenet_v2 --freeze_base --augment \
    --epochs 15 --batch_size 64 --name exp_mnv2

- Using explicit train/val directories and export TFLite (float32):
  python coral/train_tf.py \
    --train_dir data/my_ds/train --val_dir data/my_ds/val \
    --export_tflite --name exp_tflite

- Export int8 TFLite (requires representative dataset samples):
  python coral/train_tf.py \
    --dataset data/images \
    --quantize_int8 --rep_samples 200 --name exp_int8

Notes

- Default image size is 224x224. Adjust via --image_size H W.
- When using pretrained backbones (MobileNetV2 / EfficientNetB0), if weights download fails (no internet), the script falls back to random initialization.
- Outputs are written to runs_tf/<name>/, including keras models, SavedModel, labels.txt, and optional TFLite models.

