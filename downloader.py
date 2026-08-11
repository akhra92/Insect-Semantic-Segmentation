import os
import zipfile
from glob import glob

class Downloader:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.available_datasets = {
            "insect_semantic_segmentation": "kaggle datasets download -d killa92/arthropodia-semantic-segmentation-dataset",
            "plant_semantic_segmentation": "kaggle datasets download -d killa92/plant-semantic-segmentation"}
        
    def download_dataset(self, dataset_name):
        assert dataset_name in self.available_datasets, f"Dataset '{dataset_name}' not found. Available datasets: {list(self.available_datasets)}"

        url = self.available_datasets[dataset_name]
        dataset_path = os.path.join(self.save_dir, dataset_name)
        if os.path.isdir(dataset_path):
            print(f"Dataset '{dataset_name}' already exists at '{dataset_path}'. Skipping download.")
            return
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        os.system(f"{url} -p {dataset_path}")
        zip_files = glob(os.path.join(self.save_dir, dataset_name, '*.zip'))
        if not zip_files:
            raise RuntimeError(
                f"No archive was downloaded into '{dataset_path}'. Check that the Kaggle CLI is "
                "installed (`pip install kaggle`) and that ~/.kaggle/kaggle.json holds valid credentials."
            )
        # zipfile instead of the `unzip` binary, which does not exist on Windows
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file) as archive:
                archive.extractall(dataset_path)
            os.remove(zip_file)  # Remove the zip file after extraction
        print(f"Dataset '{dataset_name}' downloaded and extracted to '{dataset_path}'")

if __name__ == "__main__":
    downloader = Downloader(save_dir='datasets')
    downloader.download_dataset('insect_semantic_segmentation')