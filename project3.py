#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:38:22 2021

@author: weiweitao
"""

from pyspark.sql import SparkSession 
import pyspark.sql.functions as F
from pyspark.sql.functions import to_timestamp,date_format, lit, col, udf, substring, count
from pyspark.sql.types import DateType, IntegerType, DoubleType
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import RandomForestRegressor

spark = SparkSession.builder.config("spark.driver.memory", "16g").config("spark.executor.memory", "16g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g").appName("Project3").getOrCreate()

filename = "/Users/weiweitao/Desktop/Stony Brook/Fall 2021 Courses/AMS598/project 3/Train_project3.csv"
df_train = spark.read.options(header=True, inferSchema=True).csv(filename)

filename1 = "/Users/weiweitao/Desktop/Stony Brook/Fall 2021 Courses/AMS598/project 3/Test_project3.csv"
df_test = spark.read.options(header=True, inferSchema=True).csv(filename1)

# add the label column to df_test
df_test = df_test.withColumn("SalePrice", lit(-1))

df = df_train.unionByName(df_test)
print(df_train.count(), len(df_train.columns))
print(df_test.count(), len(df_test.columns))
print(df.count(), len(df.columns))


#------------ Part1 - deal with missing values
#fill missing values of MachineHoursCurrentMeter with 0

df = df.withColumn('MachineHoursCurrentMeter', 
                   F.when(F.col('MachineHoursCurrentMeter').contains('None') | 
                     F.col('MachineHoursCurrentMeter').contains('NULL') | 
                      F.col('MachineHoursCurrentMeter').contains('NA') | 
                     (F.col('MachineHoursCurrentMeter') == '' ) | 
                     F.col('MachineHoursCurrentMeter').isNull() | 
                     F.isnan('MachineHoursCurrentMeter'), '0').otherwise(F.col('MachineHoursCurrentMeter')))

missing = {}
for c in df.columns:
   missing[c] = df.where(F.col(c).contains('None') | 
                                 F.col(c).contains('NULL') | 
                                (F.col(c) == '' ) | 
                                F.col(c).isNull() | 
                                F.isnan(c)).count()/df.count()
   print(c, missing[c])


# drop columns with more than 80% missing
drop_cols =  [k for k,v in missing.items() if v>0.8]
df1 = df.drop(*drop_cols)

impute_cols = [k for k,v in missing.items() if (v<=0.8 and v >0)]

for c in impute_cols:
    df1 = df1.withColumn(c, F.when(F.col(c).contains('None') | 
                                F.col(c).contains('NULL') | 
                                (F.col(c) == '' ) | 
                                F.col(c).isNull() | 
                                F.isnan(c), 'Unknown').otherwise(F.col(c)))
    
    
##----------------- work on sale year, date and month
df1 = df1.withColumn("saleyear", F.split(F.col("saledate"),"[/ ]").getItem(2))
df1 = df1.withColumn("salemonth", F.split(F.col("saledate"),"[/ ]").getItem(0))
df1 = df1.withColumn("saleday", F.split(F.col("saledate"),"[/ ]").getItem(1))
df1.select('saleyear', 'salemonth', 'saleday').show()

# convert columns to numerical
df1 = df1.withColumn("saleyear", df1["saleyear"].cast(IntegerType()))
df1 = df1.withColumn("salemonth", df1["salemonth"].cast(IntegerType()))
df1 = df1.withColumn("saleday", df1["saleday"].cast(IntegerType()))
df1 = df1.withColumn("MachineHoursCurrentMeter", df1["MachineHoursCurrentMeter"].cast(DoubleType()))

df1 = df1.withColumn("YearMade", df1["YearMade"].cast(IntegerType()))

# drop following columns: saledate, SalesID, MachineID, datasource, saleyear
drop_cols1 = ['saledate', 'SalesID', 'MachineID', 'datasource']
df1 = df1.drop(*drop_cols1)


#------------------- check summary statistics of each columns
stringlist = [item[0] for item in df1.dtypes if item[1].startswith('string')]
numlist = [item[0] for item in df1.dtypes if item[1].startswith('int') | item[1].startswith('double') ]

# string columns
for c in stringlist:
    print(c + ": " + str(df1.select(c).distinct().count()))
    
# following columns have a lot different levels: fiModelDesc, fiBaseModel, fiSecondaryDesc, 
drop_cols2 = ['fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc']
df1 = df1.drop(*drop_cols2)
    
# numerical columns
num_summary = df1.select(numlist).describe().toPandas()

# find out percentiles of year made to deal with data issue with year made
percentiles = np.array(range(0, 20, 5))
np.percentile(df1.select('YearMade').collect(), percentiles)

# if year made is 1000, impute with 1966
df1 = df1.withColumn('YearMade', F.when((F.col("YearMade") < 1966), 1966).otherwise(F.col("YearMade")))
# if saleyear < year made
df1 = df1.withColumn('saleyear', F.when((F.col("saleyear") < F.col("YearMade")), F.col("YearMade")).otherwise(F.col("saleyear")))

df1.select('YearMade', 'saleyear').describe().toPandas()

# value counts for each columns
stringlist = [item[0] for item in df1.dtypes if item[1].startswith('string')]
# string columns
for c in stringlist:
   df1.groupBy(c).count().show(100)

# --------------- feature preprocessing for categorical variables with too many levels
# deal with state columns
df1 = df1.withColumn('state', F.when((F.col("state").contains('Track')|
                                          F.col("state").contains('Dozer')), 
                                         'Unknown').otherwise(F.col("state")))


df1 = df1.withColumn('state', F.when((F.col("state").contains('Track')|
                                          F.col("state").contains('Dozer')), 
                                         'Unknown').otherwise(F.col("state")))

other_state = [row[0] for row in df1.groupBy("state").agg(count('*').alias("cnt")).where(col("cnt")< 2000).select('state').collect()]

  
df1 = df1.withColumn('state', F.when(F.col("state").isin(other_state), 
                                         'Other').otherwise(F.col("state")))



#### deal with ProductGroup
other_ProductGroup = [row[0] for row in df1.groupBy("ProductGroup").agg(count('*').alias("cnt")).where(col("cnt")< 2000).select('ProductGroup').collect()]

  
df1 = df1.withColumn('ProductGroup', F.when(F.col("ProductGroup").isin(other_ProductGroup), 
                                         'Other').otherwise(F.col("ProductGroup")))


# deal with fiProductClassDesc remove tokens, numbers
df1 = df1.withColumn('fiProductClassDesc', F.regexp_replace('fiProductClassDesc', r'[0-9]*', ""))
df1 = df1.withColumn('fiProductClassDesc', F.regexp_replace('fiProductClassDesc', r'[-.+]*', ""))
df1 = df1.withColumn('fiProductClassDesc', F.regexp_replace('fiProductClassDesc', ' ', ""))


#--------------------------- string indexing for categorical columns
# create object of StringIndexer class and specify input and output column
for c in stringlist:
    oc = c+"_si"
    SI = StringIndexer(inputCol=c, outputCol=oc)
    df1 = SI.fit(df1).transform(df1)


# ----------------- split dataset into train and test
df_train = df1.filter(F.col('SalePrice') > -1)

df_test = df1.filter(F.col('SalePrice') == -1)
print(df_train.count(), len(df_train.columns))
print(df_test.count(), len(df_test.columns))


# for debug
# missing = {}
# for c in df_train.columns:
#    missing[c] = df_train.where(F.col(c).contains('None') | 
#                                  F.col(c).contains('NULL') | 
#                                 (F.col(c) == '' ) | 
#                                 F.col(c).isNull() | 
#                                 F.isnan(c)).count()/df.count()
#    print(c, missing[c])
   
   
# ------------------- vector assembler

# using vector assembler to combine all features
# specify the input and output columns of the vector assembler
features = ['ModelID',
            'YearMade',
            'MachineHoursCurrentMeter',
            'saleyear',
            'salemonth',
            'saleday',
            'auctioneerID_si',
            'ProductSize_si',
            'fiProductClassDesc_si',
            'state_si',
            'ProductGroup_si',
            'ProductGroupDesc_si',
            'Drive_System_si',
            'Enclosure_si',
            'Forks_si',
            'Ride_Control_si',
            'Transmission_si',
            'Turbocharged_si',
            'Hydraulics_si',
            'Pushblock_si',
            'Undercarriage_Pad_Width_si',
            'Backhoe_Mounting_si']

assembler = VectorAssembler(inputCols = features,
                           outputCol = 'features')

(trainingData, testData) = df_train.randomSplit([0.8, 0.2])
trainingData1 = assembler.transform(trainingData)
trainingData1 = trainingData1.select("features",trainingData.SalePrice.alias('label'))


testData1 = assembler.transform(testData)
testData1 = testData1.select("features",testData.SalePrice.alias('label'))

# --------------------------- Random Forest Regression

maxDepth = [10, 20]
maxTrees = [20]

for d in maxDepth:
    for t in maxTrees:
        rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'label',
                                   maxDepth=d, numTrees=t)
        rfModel = rf.fit(trainingData1)
        predictions = rfModel.transform(testData1)
        
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")


        rmse = evaluator.evaluate(predictions)
        print("maxDepth: {}, numTrees: {}, rmse: {}".format(d, t, rmse))


df_test1 = assembler.transform(df_test)
predictions = rfModel.transform(df_test1)
drop_cols = ['SalePrice',  'auctioneerID_si',
 'ProductSize_si',
 'fiProductClassDesc_si',
 'state_si',
 'ProductGroup_si',
 'ProductGroupDesc_si',
 'Drive_System_si',
 'Enclosure_si',
 'Forks_si',
 'Ride_Control_si',
 'Transmission_si',
 'Turbocharged_si',
 'Hydraulics_si',
 'Pushblock_si',
 'Undercarriage_Pad_Width_si',
 'Backhoe_Mounting_si',
 'features']
predictions = predictions.drop(*drop_cols)

output = "/Users/weiweitao/Desktop/Stony Brook/Fall 2021 Courses/AMS598/project 3/"
predictions.toPandas().to_csv(output+'predictions_on_testing.csv')



rfModel.featureImportances

from itertools import chain
attrs = sorted(
    (attr["idx"], attr["name"]) for attr in (chain(*trainingData1
        .schema["features"]
        .metadata["ml_attr"]["attrs"].values())))


feature_importance = [(name, rfModel.featureImportances[idx])
 for idx, name in attrs
 if rfModel.featureImportances[idx]]

feature_importance.sort(key = lambda x: x[1], reverse= False)

x = [x[0] for x in feature_importance]
y = [x[1] for x in feature_importance]
plt.figure(figsize=(12, 8))
plt.barh(x, y)
plt.xlabel("Random Forest Feature Importance")
plt.savefig('/Users/weiweitao/Desktop/Stony Brook/Fall 2021 Courses/AMS598/project 3/feature_importance.png', bbox_inches='tight', pad_inches=0)



# rf = RandomForestRegressor(labelCol="label", featuresCol="features")
# pipeline = Pipeline(stages=[rf])

# # Hyperparameter tuning
# paramGrid = ParamGridBuilder() \
#     .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 2)]) \
#     .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 2)]) \
#     .build()

# crossval = CrossValidator(estimator=pipeline,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=RegressionEvaluator(),
#                           numFolds=3)

# cvModel = crossval.fit(trainingData1)

# predictions = cvModel.transform(testData1)

# # Evaluate
# evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# rmse = evaluator.evaluate(predictions)

# rfPred = cvModel.transform(df)

# rfResult = rfPred.toPandas()

# plt.plot(rfResult.SalePrice, rfResult.prediction, 'bo')
# plt.xlabel('Price')
# plt.ylabel('Prediction')
# plt.suptitle("Model Performance RMSE: %f" % rmse)
# plt.show()