- to install requirment use this command in venv:pip install -r requirements.txt

📦image_captioning_with_vmamba
┣ 📂data <- dataset
┣ 📂notebooks <- notebooks for model prototype
┃ ┗ 📜Data_exploration_for_Image_captioning_model.ipynb <- data exploration notebook
┣ 📂src  
┃ ┣ 📂data <- scripts to download and transform data
┃ ┃ ┣ 📜build_features.py <- transform data script
┃ ┃ ┗ 📜make_data.py <- download data script
┃ ┣ 📂models <- src for model architecture + model training + model inference
┃ ┃ ┣ 📜decoder.py <- decoder of the model
┃ ┃ ┣ 📜encoder.py <- encoder of the model
┃ ┃ ┣ 📜predict.py <- inference script
┃ ┃ ┗ 📜train.py <- training script
┃ ┗ 📜**init**.py
┣ 📜pyproject.toml
┗ 📜README.md
┗ 📜requirements.txt
