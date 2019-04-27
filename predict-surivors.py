from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Titanic").getOrCreate()

df = spark.read.option("sep",",").csv("data/titanic.csv",inferSchema = "True", header = "True")

# Selecting Columns necesary for creating the model
my_cols = df.select(["Survived","PClass","Sex","Age","SibSp","Parch","Fare","Embarked"])

# Dropping rows with missing data
my_final_data = my_cols.na.drop()

# Converting categorical variables into numerical variables uning StringIndexer
# and OneHotEncoder to convert the strings into numbers

gender_indexer = StringIndexer(inputCol = "Sex", outputCol = "SexIndex")
gender_encoder = OneHotEncoder(inputCol = "SexIndex", outputCol = "SexVec")

embark_indexer = StringIndexer(inputCol = "Embarked", outputCol = "EmbarkIndex")
embark_encoder = OneHotEncoder(inputCol = "EmbarkIndex", outputCol="EmbarkVec")

# Transforming all the features into vectors using VectorAssembler
assembler = VectorAssembler(inputCols = ["PClass","SexVec","EmbarkVec","Age","SibSp","Parch","Fare"],outputCol = "features")

# Creating a Logistic Regression Object
log_reg_titanic = LogisticRegression(featuresCol = "features", labelCol = "Survived")

# Creating stages to pass to the Pipeline
pipeline = Pipeline(stages = [gender_indexer, embark_indexer, gender_encoder, embark_encoder, assembler, log_reg_titanic])

# Splitting the data randomly into test and train
train_data , test_data = my_final_data.randomSplit([0.7, 0.3])

# Fit_model will automatically create and label the predicted columm as predict
fit_model = pipeline.fit(train_data)
results = fit_model.transform(test_data)

# Calling the Binary Classification Evalutator
my_eval = BinaryClassificationEvaluator(rawPredictionCol = "prediction", labelCol="Survived")

results.select("Survived","prediction").show(100)

AUC = my_eval.evaluate(results)

print(AUC)

