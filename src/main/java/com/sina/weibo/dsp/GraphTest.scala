package com.sina.weibo.dsp

import com.tencent.angel.sona.core.DriverContext
import com.tencent.angel.sona.graph.embedding.Param
import com.tencent.angel.sona.graph.embedding.line.LINEModel
import com.tencent.angel.sona.graph.utils.{Features, SparkUtils, SubSampling}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SparkUtil

import scala.util.Random

object GraphTest {
    def main(args: Array[String]):Unit={
        val params = SparkUtil.parse(args)
        // spark和angel的启动方式，这个写法才是正确的，其他example里的写法不能正常启动
        val spark = SparkSession.builder()
            .appName("AngelDSPTest")
            .getOrCreate()
        val sparkConf = spark.sparkContext.getConf
        val driverCtx = DriverContext.get(sparkConf)
        println("graphtestargs=", params)
        driverCtx.startAngelAndPSAgent()

        val input = params.getOrElse("input", null)
        val output = params.getOrElse("output", "")
        val embeddingDim = params.getOrElse("embedding", "10").toInt
        val numNegSamples = params.getOrElse("negative", "5").toInt
        val numEpoch = params.getOrElse("epoch", "10").toInt

        val stepSize = params.getOrElse("stepSize", "0.1").toFloat
        val batchSize = params.getOrElse("batchSize", "10000").toInt
        val numPartitions = params.getOrElse("numParts", "10").toInt
        val withSubSample = params.getOrElse("subSample", "true").toBoolean
        val withRemapping = params.getOrElse("remapping", "true").toBoolean
        val order = params.get("order").fold(2)(_.toInt)
        val checkpointInterval = params.getOrElse("interval", "10").toInt
        val numCores = SparkUtils.getNumCores(sparkConf)
        // the number of partition is more than the cores.
        // we do this to achieve dynamic load balance
        val numDataPartitions = (numCores * 6.25).toInt
        println(s"numDataPartitions=$numDataPartitions")

        val data = spark.sparkContext.textFile(input)
        data.persist(StorageLevel.DISK_ONLY)
        var corpus: RDD[Array[Int]] = null
        if (withRemapping) {
            val temp = Features.corpusStringToInt(data)
            corpus = temp._1
            temp._2.map(f => s"${f._1}:${f._2}").saveAsTextFile(output+"/mapping")
        }else{
            corpus = Features.corpusStringToIntWithoutRemapping(data)
        }
        val (maxNodeId, docs) = if (withSubSample) {
            corpus.persist(StorageLevel.DISK_ONLY)
            val subSampleTmp = SubSampling.sampling(corpus)
            (subSampleTmp._1, subSampleTmp._2.repartition(numDataPartitions))
        }else{
            val tmp = corpus.repartition(numDataPartitions)
            (tmp.map(_.max).max().toLong + 1, tmp)
        }
        val edges = docs.map{arr => (arr(0), arr(1))}
        edges.persist(StorageLevel.DISK_ONLY)

        val numEdge = edges.count()
        println(s"numEdge=$numEdge maxNodeId=$maxNodeId")
        corpus.unpersist()
        data.unpersist()
        val param = new Param().setLearningRate(stepSize)
            .setEmbeddingDim(embeddingDim).setBatchSize(batchSize)
            .setSeed(Random.nextInt()).setNumPSPart(Some(numPartitions))
            .setNumEpoch(numEpoch).setNegSample(numNegSamples)
            .setMaxIndex(maxNodeId).setNumRowDataSet(numEdge)
            .setOrder(order).setModelCPInterval(checkpointInterval)
        val model = new LINEModel(param)
        model.train(edges, param, output+"/embedding")
        model.save(output + "/embedding", numEpoch)

        driverCtx.stopAngelAndPSAgent()
        spark.close()
    }
}
