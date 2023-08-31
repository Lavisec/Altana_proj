# Altana_proj
main1.py is a python code that predicts shipment COUNTRY_OF_ORIGIN by using a random forest classifier.

The main1.py code requires the following two data sets:
ds-project-train.csv
ds-project-validation.csv

1. Move all files and dependent functions into one folder
2. change the path to the folder where the data is saved

The classifier is using total of 5 features
1.	Sin_dayofyear was derived as the day number of the year the shipment arrived, This was converted to a sin function in order to account for the year periodicity
2.	 'PORT_num_N','VESSEL_num_N' where converted from STR to categorical number
3.	PRODUCT_CLASS_NUM – the product details of the shipment where parsed inot specific words, non-frequent and very frequent words were removed. Ansd then the text was classified into 100 clusters, the cluster number used as a feature.
4.	COUNTRY_num_N – is the COUNTRY_OF_ORIGIN converted to categorical number 

Classification results are displayed on the screen and saved in variable:

Metrics. Precision
Metrics. Recall
Metrics. Accuracy
Metrics. F1score

Feature Importance, are displayed in a figure
