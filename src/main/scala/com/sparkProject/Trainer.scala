package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** I) CHARGER LE DATASET **/
   val df: DataFrame = spark
     .read
     .option("header", true)  // Use first line of all files as header
     .option("inferSchema", "true") // Try to infer the data types of each column
     .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
     .parquet("./Data_spark/prepared_trainingset")

    /** II) TF-IDF **/
    /** On commence par le tokenizer **/
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    /** On retire les mots qui ne vehicent pas de sens (type "a", "the" etc...) **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    /** La partie TF de TF-IDF **/
    val countVec = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("cvModel")

    /** IDF **/
    val idf = new IDF()
      .setInputCol("cvModel")
      .setOutputCol("tfidf")

    /** III) Traitement des variables catégorielles **/
    val index_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val index_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** IV) VECTOR ASSEMBLER (assemblage des features en vecteurs) **/
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa",
                          "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /** V) MODEL de ML (regression logistique) **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** VI) PIPELINE **/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVec, idf,
        index_country, index_currency, assembler, lr))

    /** VII) TRAINING AND GRID-SEARCH **/

    /** Split des données **/
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))

    /** Création de la grille pour la recherche des bons hyper paramètres **/
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVec.minDF,  Array[Double](55, 75, 95))
      .build()

    /** Notre evaluator **/
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    /** Split pour la validation et emploi pipeline; evaluator et paramGrid **/
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /** Entrainement du modèle **/
    val model = trainValidationSplit.fit(training)

    /** Prédictions sur "test" **/
    val dfWithPredictions = model.transform(test)

    /** Score final : **/
    val f1Score = evaluator.evaluate(dfWithPredictions)
    println(s"F1 score = $f1Score")

    dfWithPredictions.groupBy("final_status", "predictions").count.show()
    println("All good !")

  }
}
