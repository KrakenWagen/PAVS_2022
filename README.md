# Audio-visual saliency prediction for 360◦ video via deep learning

- This repo contains the main code used in the master thesis [_Audio-visual saliency prediction for 360◦ video via deep learning_](https://zaguan.unizar.es/record/120377?ln=en) by Félix Bernal Sierra.

![diagram](https://github.com/KrakenWagen/PAVS_2022/blob/master/figs/architecture.png)

## Abstract

The interest in virtual reality (VR) has rapidly grown in recent years, being now widely available to consumers in different forms. This technology provides an unprecedented level of immersion, creating many new possibilities that could change the way people experience digital content. Understanding how users behave and interact with virtual experiences could be decisive for many different applications such as designing better virtual experiences, advanced compression techniques, or medical diagnosis.
One of the most critical areas in the study of human behaviour is visual attention. It refers to the qualities that different items have which makes them stand out and attract our attention.Despite the fact that there have been significant advances in this field in recent years, saliency prediction remains a very challenging problem due to the many factors that affect the behaviour of the observer, such as stimuli sources of different types or users having different backgrounds and emotional states. On top of that, saliency prediction for VR content is even more difficult as this form of media presents additional challenges such as distortions, users having control of the camera, or different stimuli possibly being located outside the current view of the observer.
This work proposes a novel saliency prediction solution for 360◦ video based on deep learning. Deep learning has been proven to obtain outstanding results in many different image and video tasks, including saliency prediction. Although most works in this field focus solely on visual information, the proposed model incorporates both visual and directional audio information with the objective of obtaining more accurate predictions. It uses a series of convolutional neural networks (CNNs) specially designed for VR content, and it is able to learn spatio-temporal visual and auditory features by using three-dimensional convolutions. It is the first solution to make use of directional audio without the need for a hand-crafted attention modelling technique. The proposed model is evaluated using a publicly available dataset. The results show that it outperforms previous state-of-the-art work in both quantitative and qualitative analysis. Additionally, various ablation studies are presented, supporting the decisions made during the design phase of the model.

## Citing

```
@ARTICLE{BernalSierra:120377,
  author= "Bernal Sierra, Félix",
  title= "{Audio-visual saliency prediction for 360◦ video via deep learning.}",
  year= "2022",
}
```

## Requirements

- python3

## Installation

```bash
conda env create -f environment.yml
```

## Usage

Make sure the values inside config.py are set correctly.

#### Train: Run `train_PAVS.py`

#### Predict: Run `predict_PAVS.py`

- The AEMs can be generated via [Spatial Audio Generation](https://github.com/pedro-morgado/spatialaudiogen)
