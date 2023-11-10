# SC4001 Project

## Installations

To install the required packages, run:

```bash
pip install -r requirements.txt
```

*Python version used is 3.8.*

## Project Structure

```bash
sc4001-project/
├── data/
├── models/
│   ├── saved_models/
│   │   └── data_visualisation.ipynb
│   ├── common_utils.py
│   └── gradmap_utils.py
├── notebooks/
│   ├── 0_baseline_model_preprocessed.ipynb
│   ├── 1_baseline_model.ipynb
│   ├── 2_batchnorm_model.ipynb
│   ├── 3a_batchnorm_reduced_model.ipynb
│   ├── 3b_batchnorm_reduced_model.ipynb
│   ├── 3c_batchnorm_reduced_model.ipynb
│   ├── 4_depthpointwise_model.ipynb
│   ├── 5b_onecycle_lr_model.ipynb
│   ├── 6_mixup_model.ipynb
│   ├── 7_augment_model.ipynb
│   ├── 8_final_model.ipynb
│   ├── evaluation.ipynb
│   ├── grad_map.ipynb
│   ├── test_data.ipynb
│   └── test_model.ipynb
├── src/
├── requirements.txt
└── report.pdf
```

The `data` directory contains the Oxford Flowers 102 dataset. Note that the downloading will be handled by `torchvision`(in the notebooks).

The `models` directory contains:
- `saved_models`: The saved states of all the models we have trained. Note that we use `cuda`.
- `saved_models/data_visualisation.ipynb`: Notebook for visualisation of various graphs (accuracy, loss, etc.).
- `common_utils.py`: Consists of various global helper functions.
- `gradmap_utils.py`: Consists of various helper functions for the generation of Grad-CAM in `notebooks/grad_map.ipynb`.
- `model.py`: Consists of different model architectures.

The `notebooks` directory contains notebooks for training various models, evaluating them, and visualising data and model performances.

`report.pdf` is our detailed report in the implementation of this project

### Note (For assessors only):
Trained models inside `/models/saved_models` and data in `/data` are excluded from the final submission to NTU due to space constraints.

Our trained models are available to be downloaded at our [GitHub repository](https://github.com/A-Alviento/sc4001-project) in the `/models/saved_models` directory.

For the data, run the first two cells of `/notebooks/0_baseline_model_preprocessed.ipynb` to download the dataset.

*(Ignore this section unless you are the one assessing this project)*

### Getting Started

Ensure all requirements are satisfied by running:

```bash
pip install -r requirements.txt
```

This project has been tested on python=3.8

#### Training Your Own Models

You can start training models by accessing the notebooks in the `/notebooks` directory:

- `0_baseline_model_preprocessed.ipynb`: Train the baseline model with the original saliency preprocessing technique.
- `1_baseline_model.ipynb`: Train the baseline model with standard normalisation preprocessing.
- Subsequent notebooks (`2_batchnorm_model.ipynb` to `8_final_model.ipynb`): Train models with various techniques and improvements.
- Note: the various techniques are cascaded down, so every notebook incorporates it's own new technique, as well as the techniques preceding it.
- Evaluate the models using `evaluation.ipynb`.
- Generate Grad-CAM heatmaps using `grad_map.ipynb`.
- For data visualisation, refer to `/models/saved_models/data_visualisation.ipynb`.

#### Loading Our Trained Models

Download the desired models from our [GitHub repository](https://github.com/A-Alviento/sc4001-project). Then evaluate and generate heatmaps as described above.

### Acknowledgements
[1]	Y. Liu, F. Tang, D. Zhou, Y. Meng, and W. Dong, “Flower classification via convolutional neural network,” in 2016 IEEE International Conference on Functional-Structural Plant Growth Modeling, Simulation, Visualization and Applications (FSPMA), Qingdao, China: IEEE, Nov. 2016, pp. 110–116. doi: 10.1109/FSPMA.2016.7818296. <br>
[2]	M.-E. Nilsback and A. Zisserman, “Automated Flower Classification over a Large Number of Classes,” in 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, Dec. 2008, pp. 722–729. doi: 10.1109/ICVGIP.2008.47.<br>
[3]	S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” arXiv, Mar. 02, 2015. doi: 10.48550/arXiv.1502.03167.<br>
[4]	H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond Empirical Risk Minimization.” arXiv, Apr. 27, 2018. doi: 10.48550/arXiv.1710.09412.<br>
[5]	A. Gurnani, V. Mavani, V. Gajjar, and Y. Khandhediya, “Flower Categorization using Deep Convolutional Neural Networks.” arXiv, Dec. 08, 2017. doi: 10.48550/arXiv.1708.03763.<br>
[6]	L. N. Smith and N. Topin, “Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.” arXiv, May 17, 2018. Accessed: Nov. 10, 2023. [Online]. Available: http://arxiv.org/abs/1708.07120<br>
[7]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.” Dec. 02, 2019. doi: 10.1007/s11263-019-01228-7.
