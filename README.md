# NlpSentimentAnalysis

## Introduction
Fine-grained emotional analysis of online reviews is of great value to deeply understand businesses and users and to tap users'emotions. It is widely used in the Internet industry, mainly for personalized recommendation, intelligent search, product feedback, business security and so on. This project completes the task of fine-grained emotional analysis through a high-quality massive data set, which contains six categories and 20 fine-grained elements. We need to build an algorithm based on the sentiment tendency of the annotated fine-grained elements, mine the user comments, determine the prediction accuracy by calculating the error between the predicted value and the real value of the scene, and evaluate the proposed prediction algorithm.
## Current performance

## Usage

1. Install pytorch 1.0 for Python 3.6+
2. Run `pip3 install -r requirements.txt` to install python dependencies.
3. Run `python main.py --mode data` to build tensors from the raw dataset.
4. Run `python main.py --mode train` to train the model. After training, `log/model.pt` will be generated.
5. Run `python main.py --mode test` to test an pretrained model. Default model file is `log/model.pt`

## Structure
preproc.py: downloads dataset and builds input tensors.

main.py: program entry; functions about training and testing.

models.py: The sentiment analaysis neural network structure.

config.py: configurations.

## Differences from the paper

1. The paper doesn't mention which activation function they used. I use relu.
2. I don't set the embedding of `<UNK>` trainable.
3. The connector between embedding layers and embedding encoders may be different from the implementation of Google, since the description in the paper is inconsistent (residual block can't be used because the dimensions of input and output are different) and they don't say how they implemented it.

## TODO

- [x] Reduce memory usage
- [ ] **Improve converging speed (to reach 60 F1 scores in 1000 iterations)**
- [ ] Reach state-of-art scroes of the original paper
- [ ] Performance analysis
- [ ] Test on AI-Challenger2018 dataset
