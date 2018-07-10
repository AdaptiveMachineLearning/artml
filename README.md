# Adaptive Real Time Machine Learning (ART-ML)

The term “Real Time” is used to describe how well predictive modeling algorithms can accommodate an ever increasing data load instantaneously. However, such real time problems are usually closely coupled with the fact that conventional data mining algorithms operate in a batch mode where having all of the relevant data at once is a requirement. Thus, here Real Time Machine Learning  is defined as having all of the following characteristics, independent of the amount of data involved: 

Incremental learning (Learn): immediately updating a model with each new observation without the necessity of pooling new data with old data.

Decremental learning (Forget): immediately updating a model by excluding observations identified as adversely affecting model performance without forming a new dataset omitting this data and returning to the model formulation step.

Variable addition (Grow): Adding a new attribute (variable) on the fly, without the necessity of pooling new data with old data.

Variable deletion (Shrink): immediately discontinuing use of an attribute identified as adversely affecting model performance.

Distributed processing: separately processing distributed data or segments of large data (that may be located in diverse geographic locations) and re-combining the results to obtain a single model.

Parallel processing: carrying out parallel processing extremely rapidly from multiple conventional processing units (multi-threads, multi-processors or a specialized chip).

This work is Python adoption for Prof. Sayad (University of Toronto) research on Real time Machine Learning.

http://www.saedsayad.com/data_mining_map.htm

https://www.researchgate.net/publication/265619432_Real_Time_Data_Mining


Project in PROGRESS.........
