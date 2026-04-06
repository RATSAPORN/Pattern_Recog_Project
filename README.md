- to install requirment use this command in venv: pip install -r requirements.txt
- after implement please ensure to use this command for update requirment: pip freeze > requirements.txt

```bash
image_captioning_with_vmamba/
├── data/                  # dataset
├── notebooks/             # notebooks for model prototype
│   └── Data_exploration_for_Image_captioning_model.ipynb
├── src/
│   ├── data/              # scripts to download and transform data
│   │   ├── build_features.py
│   │   └── make_data.py
│   ├── models/            # model architecture + training + inference
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── train.py
│   │   └── predict.py
│   └── __init__.py
├── pyproject.toml         # describe project
├── README.md
└── requirements.txt       # python packages
```
