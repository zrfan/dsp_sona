package com.sina.weibo.dsp

//import com.tencent.angel.ml.classification.LogisticRegression

import com.tencent.angel.sona.context.PSContext
import org.apache.spark.ml.classification.LogisticRegression
import com.tencent.angel.sona.core.DriverContext
import com.tencent.angel.sona.ml.classification.AngelClassifier
import com.tencent.angel.sona.ml.regression.AngelRegressor
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable

object SonaTest {

    def main(args: Array[String]): Unit = {
        println("args=", args)
        val params = parse(args)
        println("params=", params)
        val modelPath = params.getOrElse("modelPath","")

        val spark = SparkSession.builder()
            .appName("AngelTestSona")
            .getOrCreate()
        val sparkConf = spark.sparkContext.getConf
        val driverCtx = DriverContext.get(sparkConf)
        driverCtx.startAngelAndPSAgent()

        println("PSContext conf=", driverCtx.hadoopConf)

        val training = spark.createDataFrame(Seq(
            (1.0, Vectors.dense(0.0, 1.1, 0.1)),
            (0.0, Vectors.dense(2.0, 1.0, -1.0)),
            (0.0, Vectors.dense(2.0, 1.3, 1.0)),
            (1.0, Vectors.dense(0.0, 1.2, -0.5))
        )).toDF("label", "features")

        val lr = new LogisticRegression()
        println("LogisitcRegerssion parameters:\n" + lr.explainParams())
//        lr.setMaxIter(10).setLearningRate(0.01)
        lr.setMaxIter(10).setRegParam(0.01)
        val model1 = lr.fit(training)
        println("Model 1 was fit using parameters:" + model1.parent.extractParamMap())
        println("model path=",driverCtx.sharedConf.get("angel.save.model.path"))
        println("save model path=", modelPath)
        model1.write.overwrite().save(modelPath)

        val paramMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter, 30).put(lr.regParam -> 0.1, lr.threshold -> 0.55)

        val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
        val paramMapCombined = paramMap ++ paramMap2

        val model2 = lr.fit(training, paramMapCombined)
        println("Model 2 was fit using parameters:" + model2.parent.extractParamMap())


        //prepare test data
        val test = spark.createDataFrame(Seq(
            (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
            (0.0, Vectors.dense(3.0, 2.0, -0.1)),
            (1.0, Vectors.dense(0.0, 2.2, -1.5))
        )).toDF("label", "features")
        model2.transform(test).select("features", "label", "myProbability", "prediction").collect().foreach {
            case Row(features: DenseVector, label: Double, prob: DenseVector, prediction: Double) =>
                println(s"($features, $label) -> prob=$prob, prediction=$prediction")
        }
        // 终止PSContext
        //        PSContext.stop()
        driverCtx.stopAngelAndPSAgent()
        spark.close()

        //
        //    val libsvm = spark.read.format("libsvmex")
        //    val dummy = spark.read.format("dummy")
        //
        //    val trainData = libsvm.load("./data/angel/a9a/a9a_123d_train.libsvm")
        //
        //    val classifier = new AngelClassifier()
        //      .setModelJsonFile("./angelml/src/test/jsons/logreg.json")
        //      .setNumClass(2)
        //      .setNumBatch(10)
        //      .setMaxIter(2)
        //      .setLearningRate(0.1)
        //
        //    val model = classifier.fit(trainData)
        //    model.write.overwrite().save("trained_models/lr")
    }
    def parse(args: Array[String]): Map[String, String] = {
    val cmdArgs = new mutable.HashMap[String, String]()
    println("parsing parameter")
    for (arg <- args) {
      val KEY_VALUE_SEP = ":"
      val sepIdx = arg.indexOf(KEY_VALUE_SEP)
      if (sepIdx != -1) {
        val k = arg.substring(0, sepIdx).trim
        val v = arg.substring(sepIdx + 1).trim
        if (v != "" && v != "Nan" && v != null) {
          cmdArgs.put(k, v)
          println(s"param $k = $v")
        }
      }
    }
    cmdArgs.toMap
  }
}
