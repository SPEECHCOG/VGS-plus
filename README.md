# VGS-plus

current plan

practical steps:

1) Training the FaST-VGS and FaST-VGS+ original models and replicating Peng's results on SC data for recall and ABX error.

2) Training the two original models without using w2v2 pretrained wights and with random initialization.

3) Monitoring loss values (SSL and VGS terms), separately during training as well as recall and ABX scores.

- by observing behavior of loss terms and other metrics (e.g., recall and ABX error) during training (to their best values), we can unerstand the model behavior as a function of training progress, and check for example if phonetic and semantic knowledge appear at the same time, and go hand-to-hand or how their help each other. We can also compare the results with VGS-only and SSL-only models on same amount of data to see how one model affects other, for example how SSL helps to obtain better results on semantic tasks and how VGS helps to obtain better results on phonetic tasks. 

- we can also examine the effect of SSL model choice (for example by testing masking and predictive models), on above results.
