package com.sina.weibo.dsp

import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.sql.{Row, SparkSession}
import com.sina.weibo.dsp.embedding.NCFModel
import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ps.storage.matrix.PartitionSourceArray
import com.tencent.angel.sona.context.PSContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SparkUtil

import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable

object NCFTest {
    def main(args: Array[String]): Unit = {

        val sparkSession = SparkSession.builder().appName("NCFTest")
            //                        .config("hive.metastore.warehouse.dir", "")
            .config("spark.sql.warehouse.dir", "hdfs://ns3-backup/user/weibo_bigdata_push/weichao_spark_sql_warehouse")
            .config("hive.exec.scratchdir", "hdfs://ns3-backup/user/weibo_bigdata_push/weichao_hive_scratchdir")
            .enableHiveSupport()
            .getOrCreate()
        val sparkContext = sparkSession.sparkContext
        val sparkConf = sparkContext.getConf
        //        val sparkConf = new SparkConf()
        //        val sparkContext = new SparkContext(sparkConf)

        sparkConf.set(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS, classOf[PartitionSourceArray].getName)
        sparkConf.set(AngelConf.ANGEL_PS_BACKUP_MATRICES, "")
        sparkConf.set("angel.save.model.path", "/user/weibo_bigdata_push/zhongruix/angel_model/tmp/")
        sparkConf.set("action.type", "train")
        //        println("sparkConf_context=", sparkConf.getAll.mkString("\n"))
        //        println("spark_env_conf", SparkEnv.get.conf.getAll.mkString("\n"))
        PSContext.getOrCreate(sparkContext)
        println("args=", args.mkString(", "))
        val params = SparkUtil.parse(args)
        val dt = params.getOrElse("dt", "")
        val numPartitions = 700

        // ================== mock data ======================
        val data1 = (1 to 20).toArray
        val data2 = (20 to 35).toArray
        val data3 = (10 to 25).toArray
        val data4 = (30 to 55).toArray
        var data: List[(Int, Array[Int], Array[Int])] = List((1, data1, data2), (2, data2, data3), (3, data3, data4), (4, data4, data1))
        var trainData = sparkContext.parallelize(data)
        val itemIds = List(1, 3, 7, 10, 12)
        val testData = sparkContext.parallelize(itemIds)
        // ================= Data Process ====================

        val data_sql =
            s"""select C.uuid, collect_list(umid) as pitems
                        from
                            (select uuid, umid, act_time from
                                zhongrui3_hot_click_data where dt=$dt
                                order by uuid, act_time ASC
                            ) AS C
                        group by C.uuid limit 2100000"""
        val topSql =
            s"""select xid, click
               |from zhongrui3_xid_click
               |where dt>=$dt and type='mid'
               |order by click desc limit 500""".stripMargin

        val itemSql =
            s"""select distinct(umid) from zhongrui3_hot_click_data
               |where dt=$dt """.stripMargin
        println("data_sql=", data_sql)

        val itemData = sparkSession.sql(itemSql).rdd.map(p => p(0)).collect()
        println("itemData=", itemData.slice(0, 10).mkString(", "))
        val bitems = sparkContext.broadcast(itemData.toSet)
        val allData = sparkSession.sql(data_sql).rdd
                                .repartition(numPartitions).map(p => (p(0).asInstanceOf[Int],
                                                    p(1).asInstanceOf[mutable.WrappedArray[Int]].toSet,
                                                    Random.shuffle(bitems.value).take(300).asInstanceOf[Set[Int]]))
                                .map(p => (p._1, p._2.toArray, (p._3 &~ p._2).toArray))
                                .filter(p => p._2.length > 5).coalesce(numPartitions)
                                .persist(StorageLevel.MEMORY_AND_DISK)
        println("allData_partitions=", allData.getNumPartitions)
        //        allData.take(5).foreach(p => println(p._1 + ", ", p._2.mkString(", ") + ", " +p._3.mkString(", ")))

        val test_Data = allData.map(p => (p._1, p._2.slice(p._2.length - 1, p._2.length),
                                            p._3.slice(p._3.length - 1, p._3.length)))
                                .filter(p => p._2.length > 0)
                                .persist(StorageLevel.MEMORY_AND_DISK)
        println("testData_partitions=", test_Data.getNumPartitions)
//        test_Data.take(1).foreach(p => println(p._1 + ", ", p._2.mkString(", ") + ", " + p._3.mkString(", ")))

        // 数据采样，每个用户只保留50个样本
        val train_Data = allData.map(p => (p._1,
                                        p._2.slice(p._2.length - 51, p._2.length - 1), p._3.slice(0, 1)))
                                .filter(p => p._2.length > 1)
                                .persist(StorageLevel.MEMORY_AND_DISK)
        println("trainData_partitions=", train_Data.getNumPartitions)
//        train_Data.take(1).foreach(p => println(p._1 + ", ", p._2.mkString(", ") + ", " + p._3.mkString(", ")))

        val topMid = sparkSession.sql(topSql).rdd.map(p => p(0).toString)
            //                            .filter(p => (System.currentTimeMillis() - (p.toInt >> 22) + 515483463) < 8*24*60*60)
            .collect().mkString(",")
        println("topSql=", topSql)
        println("get_top_mids=", topMid)
        allData.unpersist()

        val topItemSql = s"""select distinct(umid) from zhongrui3_hot_click_data where dt=$dt and mid in (${topMid})"""
        println("topItemSql=", topItemSql)
        val predictData = sparkSession.sql(topItemSql).rdd.map(p => p(0).toString.toInt)

        println("predictData=", predictData.collect().mkString(", "))

        // ======================  model train ============================
        try {
            val model = new NCFModel(7500000, 1500000, 10,
                100000, 10, 100)
            model.train(train_Data, 1, 0.0001, 20, 1,
                "/user/weibo_bigdata_push/zhongruix/angel_tmp/")
            // 前日高点击mid
            val result = model.predict(predictData, 10)
            for (arr <- result) {
                println(arr._1, " predictResult=", arr._2.mkString(", "))
            }
            //结果转换
            val resIds = result.flatMap(f=>f._2.map(x=>x._1))
            println("resIds=", resIds.mkString(", "))
            val resSql = s"""select distinct(mid),0 from zhongrui3_hot_click_data where dt=$dt and umid in (${resIds.mkString(",")})"""
            val resMid = sparkSession.sql(resSql)
            println("resMids=", resMid.rdd.collect().mkString(", "))
            resMid.createTempView("zhongrui3_ncf_temp")
            sparkSession.sql(s"insert overwrite table default.zhongrui3_xid_click partition(dt=$dt, type='ncf-pre') select * from zhongrui3_ncf_temp")
        }catch {
            case e:Throwable => println("train error! ")
        }

        train_Data.unpersist()
        test_Data.unpersist()
        PSContext.stop()
        sparkContext.stop()
    }
}
