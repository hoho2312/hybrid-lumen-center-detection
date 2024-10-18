# hybrid-lumen-center-detection
A lumen center estimation method based on deep learning (modified LDC model) and traditional CV techniques

Paper has been submitted to IEEE TIM with title: Robust Colon Lumen Center Measurement Enables Autonomous Navigation of an Electromagnetically Actuated Soft-Tethered Colonoscope

This repo will consist of the trained modified LDC model for lumen contour detection, the code for lumen center estimation (which is the algorithm proposed in the submitted paper). Dataset for training and testing will be uploaded soon.

For the modified LCD model, we propose a UpDecoder module which effectively aggregate encoder features from all resolution level from the original LDC backbone, leading to significant increase in both ODS and OIS F-scores(+4.3% and +7.8% respectively), as well as in average precision (+11.7%), while keeping a real-time performance (46.67FPS on RTX3060Ti).

<!--The lumen contour dataset (created from randomly selected subset images of LDPolypVideo dataset) can be found on: [onedrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155079256_link_cuhk_edu_hk/EltFmbpMGAlFgwFRzkocLKwBwnSEk3fXOf43bCOlWVl2hA?e=w1kVVq)-->

To train and test the LCD model, please refer to the repo of LCD or DexiNed for further reading.

Dataset: [Onedrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155079256_link_cuhk_edu_hk/EjOFmuLVXYJAgv5DvGOhUy8Bs_P8DMoMoM3FeHiGnQFlyw?e=NA8xNI)
