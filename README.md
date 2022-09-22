# VGS-plus

current plan

practical steps:

1) Training the FaST-VGS and FaST-VGS+ original models and replicating Peng's results on SC data for recall and ABX error.(model 1, model2)

2) Adding a function for saving intermediate recall and loss values during training.

3) Adding a function to save model weights based on a user-defined step argument (n_save_model) to be used later for other evaluation metrics. 

4) Training the two original models without using w2v2 pretrained weights and with random initialization. (model 3, model 4)

5) Testing different coefficients for w2v2 loss and vgs loss. (model 5, model 6)

6) Monitoring loss values (SSL and VGS terms), separately during training as well as recall and ABX scores for above models and reporting the graphs.

- by observing behavior of loss terms and other metrics (e.g., recall and ABX error) during training (for example as percentage of their best (final) values), we can unerstand the model behavior as a function of training progress, and check for example if phonetic and semantic knowledge appear at the same time, and go hand-to-hand or how their help each other. We can also compare the results with VGS-only and SSL-only models on same amount of data to see how one model affects other, for example how SSL helps to obtain better results on semantic tasks and how VGS helps to obtain better results on phonetic tasks. 

- we can also examine the effect of SSL model choice on above results (for example by comparing masking and predictive models).

- we can also examine the model behavior while changing from one dataset to another during training, for example from SC and to Flickr; in normal case when training is happening within one modality only, domain shift might has a big negative effect on training, however, when SSL is accompanied by (coupled with) semantic visual mapping, the effect of domain shift might be reduced due to semantic grounding.
