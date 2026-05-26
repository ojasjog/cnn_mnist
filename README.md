---

### Dataset Download Instructions

The EMNIST dataset is hosted on Kaggle. This repository is configured to read the **EMNIST Balanced** split.

To download the dataset automatically via the terminal without using a web browser, install the official `kagglehub` utility package:

```bash
pip install kagglehub pandas

```

Then, run this quick Python script or one-liner to download the data to your local machine's unified cache:

```python
import kagglehub

# Download latest version of the EMNIST dataset
path = kagglehub.dataset_download("crawford/emnist")
print("Dataset downloaded to:", path)

```

#### Moving the Data to Your Workspace

1. Navigate to the path printed by the script above.
2. Locate the file named `emnist-balanced-train.csv`.
3. Copy or move `emnist-balanced-train.csv` directly into the root folder of this cloned repository.

