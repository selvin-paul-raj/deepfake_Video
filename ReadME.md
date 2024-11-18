## Installation and Dataset Download

###  Install Requirements
Make sure you have Python 3.8 or later installed. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

### Step 2: Download the Dataset
Download the [Deepfake Detection Challenge (DFDC) Dataset](https://www.kaggle.com/datasets/emmanuelpintelas/dfdc-facial-cropped-videos-dataset-jpg-frames) from Kaggle. Follow these steps:

1. **Sign in to Kaggle**: 
   If you don't already have a Kaggle account, create one at [https://www.kaggle.com](https://www.kaggle.com).

2. **Download Kaggle API Key**:
   - Go to your Kaggle account settings: [Account Settings](https://www.kaggle.com/account).
   - Scroll to the *API* section and click **Create New API Token**.
   - A `kaggle.json` file will be downloaded.

3. **Place the API Key**:
   Place the `kaggle.json` file in the `.kaggle` directory in your home folder:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

4. **Install Kaggle CLI**:
   Install the Kaggle command-line tool:

   ```bash
   pip install kaggle
   ```

5. **Download the Dataset**:
   Run the following code to download the dataset:

   ```python
    import kagglehub 
    dfdc_facial_cropped_videos_dataset_jpg_frames_path = kagglehub.dataset_download('emmanuelpintelas/dfdc-facial-cropped-videos-dataset-jpg-frames')

   ```


