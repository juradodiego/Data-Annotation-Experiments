## Eliminating Bias in AI: Compensating for Label Bias in Data Sets through Re-Annotation
### [Insert Link to Paper Here]<br/>

Diego Jurado<br/>
University of Pittsburgh - Dept. of Computer Science<br/>
CS 1699 / 1951 - Spring 2022<br/>
Term Research Project<br/>

Abstract: <br/> The data sets that are most widely used for automatic hate speech detection contain Tweets or other social media-based messages, that are manually judged and annotated by largegroups of people. These annotators are human, they are prone to bias, we investigate the task of modifying manually-annotated data with a given rule set, and observe how it influences a classification task using Support Vector Machines(SVM). We first re-annotate randomly, then we introduce an intelligent method, and compare the performance of these two implementations to a baseline binary classification SVM, arguing that the closer a given incorrect prediction is to the decision boundary of an SVM, the more that Tweet or message was originally incorrectly annotated.

Experimental Results:<br/>
Precision:
- Avg Baseline Change:    +0.00000
- Avg Random Change:      −0.08712
- Avg Intelligent Change: +0.03653

Recall:
- Avg Baseline Change:    +0.00000
- Avg Random Change:      −0.08722
- Avg Intelligent Change: +0.03700

F1 Score:
- Avg Baseline Change:    +0.00000
- Avg Random Change:      −0.08744
- Avg Intelligent Change: +0.03698

