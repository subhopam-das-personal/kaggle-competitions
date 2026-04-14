#!/bin/bash
# Downloads transformers>=5.3.0 wheels for offline Kaggle use.
# Run once locally, then upload the wheels/ folder to Kaggle as a dataset.
#
# Usage:
#   chmod +x download_offline_wheels.sh
#   ./download_offline_wheels.sh
#
# Then:
#   kaggle datasets create -p wheels/ --dir-mode tar
#   (or upload via kaggle.com/datasets/new)

set -e

WHEELS_DIR="$(dirname "$0")/../wheels"
mkdir -p "$WHEELS_DIR"

echo "Downloading transformers>=5.3.0 wheels (Python 3.12, Linux x86_64)..."
echo "Output: $WHEELS_DIR"
echo ""

pip download \
  "transformers>=5.3.0" \
  "tokenizers>=0.19" \
  --dest "$WHEELS_DIR" \
  --python-version 3.12 \
  --platform manylinux2014_x86_64 \
  --only-binary :all: \
  --quiet

echo ""
echo "Downloaded files:"
ls -lh "$WHEELS_DIR"
echo ""
echo "Total size: $(du -sh "$WHEELS_DIR" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Create Kaggle dataset from the wheels/ folder:"
echo "     kaggle datasets init -p $WHEELS_DIR"
echo "     # Edit $WHEELS_DIR/dataset-metadata.json:"
echo "     #   set 'title' to: nemotron-transformers-wheels"
echo "     #   set 'id' to:    \$(kaggle config view | grep username)/nemotron-transformers-wheels"
echo "     kaggle datasets create -p $WHEELS_DIR --dir-mode tar"
echo ""
echo "  2. Add the dataset as notebook input in Kaggle:"
echo "     Notebook → Add Input → Datasets → search 'nemotron-transformers-wheels'"
echo ""
echo "  3. The notebook will auto-detect wheels at:"
echo "     /kaggle/input/nemotron-transformers-wheels/"
