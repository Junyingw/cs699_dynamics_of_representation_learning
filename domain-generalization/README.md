# CSCI 699 Project 3: Data Shift & Task Dynamics
In this project we explored two novel approaches of domain adaptiona and domain generation.<br/>
The references papaers are:<br/>
[Adversarial Discriminative Domain Adaptation (ADDA)](https://arxiv.org/pdf/1702.05464.pdf)<br/>
[Invariant Risk Minimization (IRM)](https://arxiv.org/abs/1907.02893)<br/>
Here we use the base code from [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)



# Usage
* Installation
```
python setup.py install
```
* Train & Test
```
cd ./examples/domain_adaptation/adda
sh adda.sh
```
```
cd ./examples/domain_generalization/irm
sh irm.sh
```
# Result
Here we show the accuracy comparison results of these two methods on the DomainNet dataset.
|   | Train with unlabeled target domain  |  ACC |
| :----: | :----: | :----: |
| ADDA |    &check; | 62.188 |
|  IRM |    &cross; | 58.819 |
