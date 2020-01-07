package com.sina.weibo.dsp

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ps.storage.matrix.PartitionSourceArray
import com.tencent.angel.sona.context.PSContext
import com.tencent.angel.sona.core.DriverContext
import com.tencent.angel.sona.graph.embedding.Param
import com.tencent.angel.sona.graph.embedding.word2vec.Word2VecModel
import com.tencent.angel.sona.graph.utils.{Features, SparkUtils, SubSampling}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.util.SparkUtil
import org.codehaus.jackson.JsonParser.Feature

import scala.util.Random

object W2vTest {

    def main(args: Array[String]):Unit={
        val params = SparkUtil.parse(args)
        val conf = new SparkConf()
        val sc   = new SparkContext(conf)

        conf.set(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS, classOf[PartitionSourceArray].getName)
        conf.set(AngelConf.ANGEL_PS_BACKUP_MATRICES, "")

        PSContext.getOrCreate(sc)

        val input = params.getOrElse("input", "")
        val output = params.getOrElse("output", "")
        val embeddingDim = params.getOrElse("embedding", "10").toInt
        val numNegSamples = params.getOrElse("negative", "5").toInt
        val windowSize = params.getOrElse("window", "10").toInt
        val numEpoch = params.getOrElse("epoch", "10").toInt
        val stepSize = params.getOrElse("stepSize", "0.1").toFloat
        val batchSize = params.getOrElse("batchSize", "10000").toInt
        val numPartitions = params.getOrElse("numParts", "10").toInt
        val withSubSample = params.getOrElse("subSample", "true").toBoolean
        val withRemapping = params.getOrElse("remapping", "true").toBoolean
        val modelType = params.getOrElse("modelType", "cbow")
        val checkpointInterval = params.getOrElse("interval", "10").toInt
        val numCores = SparkUtils.getNumCores(conf)
        val numDataPartitions = (numCores * 6.25).toInt
//        val data = sc.textFile(input)
//        data.persist(StorageLevel.DISK_ONLY)
//        var corpus: RDD[Array[Int]] = null
//        var denseToString: Option[RDD[(Int, String)]] = None
//        if (withRemapping){
//            val temp = Features.corpusStringToInt(data)
//            corpus = temp._1
//            denseToString = Some(temp._2)
//        }else{
//            corpus = Features.corpusStringToIntWithoutRemapping(data)
//        }
//        val (maxWordId, docs) = if (withSubSample) {
//            corpus.persist(StorageLevel.DISK_ONLY)
//            val subsampleTmp = SubSampling.sampling(corpus)
//            (subsampleTmp._1, subsampleTmp._2.repartition(numDataPartitions))
//        } else{
//            val tmp = corpus.repartition(numPartitions)
//            (tmp.map(_.max).max().toLong + 1, tmp)
//        }
//        docs.persist(StorageLevel.DISK_ONLY)
//
//        val numDocs = docs.count()
//        val numTokens = docs.map(_.length).sum().toLong
//        val maxLength = docs.map(_.length).max()
//        println(s"numDocs=$numDocs maxWordId=$maxWordId " +
//            s"numTokens=$numTokens maxLength=$maxLength")
//        corpus.unpersist()
//        data.unpersist()
        val numDocs = 10
        val maxWordId = 1000
        val maxLength = 10
        val arr1 = ((1 to 50).toArray)
        val arr2 = ((30 to 100).toArray)
        val docs = sc.parallelize(arr2)

        val param = new Param().setLearningRate(stepSize)
            .setEmbeddingDim(embeddingDim).setWindowSize(windowSize)
            .setBatchSize(batchSize).setSeed(Random.nextInt())
            .setNumPSPart(Some(numPartitions)).setNumEpoch(numEpoch)
            .setNegSample(numNegSamples).setMaxIndex(maxWordId)
            .setNumRowDataSet(numDocs).setMaxLength(maxLength)
            .setModel(modelType).setModelCPInterval(checkpointInterval)
        val model = new Word2VecModel(param)
//        model.train(docs, param, output+"/embedding")
//        denseToString.map(rdd => rdd.map(f => s"${f._1}: ${f._2}").saveAsTextFile(output + "/mapping"))
        PSContext.stop()
        sc.stop()
    }

}
