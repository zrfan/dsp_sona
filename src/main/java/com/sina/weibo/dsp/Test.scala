package com.sina.weibo.dsp
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

class Test {
  def main(args: Array[String]):Unit = {
    println("hello world")
    val spark = getSparkSession()
    import spark.implicits._
    val test = spark.sparkContext.parallelize(List((1,2,3), (4,5,6))).toDF("a", "b", "c")
    test.show(5, false)
  }

  def getSparkSession(): SparkSession={
    val sparkConf = new SparkConf().setAppName("test")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    spark
  }
}
