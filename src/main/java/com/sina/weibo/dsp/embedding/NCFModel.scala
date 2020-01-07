package com.sina.weibo.dsp.embedding

import java.util

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.storage.{DoubleVectorStorage, IntFloatDenseVectorStorage}
import com.tencent.angel.ml.math2.utils.RowType
import com.tencent.angel.ml.matrix.psf.aggr.{Dot, Pull}
import com.tencent.angel.ml.matrix.psf.get.getrow.GetRowResult
import com.tencent.angel.ml.matrix.psf.update.{Fill, RandomNormal}
import com.tencent.angel.ps.storage.matrix.PartitionSourceArray
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.sona.context.PSContext
import com.tencent.angel.sona.models.{PSMatrix, PSVector}
import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.sona.util.VectorUtils

import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable

class NCFModel(userRow: Int, itemRow: Int, numCol: Int,
               rowsInBlock: Int, colsInBlock: Int,
               seed: Int) extends Serializable {
    def this() = this(-1, -1, -1, -1, -1, 0)

    // create new matrix
    val userPSMatrix: PSMatrix = createPSMatrix(userRow, numCol, rowsInBlock, colsInBlock)
    val userMatrixId: Int = userPSMatrix.id
    val itemPSMatrix: PSMatrix = createPSMatrix(itemRow, numCol, rowsInBlock, colsInBlock)
    val itemMatrixId: Int = itemPSMatrix.id

    val w1Matrix: PSMatrix = createPSMatrix(2, numCol, rowsInBlock, colsInBlock)
    val w1MatrixId: Int = w1Matrix.id
    val bias1: Double = 0.0

    val h1Vector: PSVector = createVector(2)

    private val rand = new Random(seed)
    // initialize embeddings
    randomInitialize(userPSMatrix, rand.nextInt())
    randomInitialize(itemPSMatrix, rand.nextInt())

    // 模型预测, 给定itemId，计算与之相似的itemID集合
    def predict(dataSet: RDD[Int], topK: Int): Array[(Int, Array[(Int, Double)])] = {
        val numPartitions = dataSet.getNumPartitions
        //        println("")
        val res = dataSet.mapPartitionsWithIndex((partitionId, iter) =>
            predictForPartition(partitionId, iter, numPartitions, topK)).collect()
        res
    }

    def predictForPartition(partitionId: Int, iter: Iterator[Int], numPartitions: Int, topK: Int): Iterator[(Int, Array[(Int, Double)])] = {
        def predictSingle(partitionId: Int, itemId: Int, topK: Int): (Int, Array[(Int, Double)]) = {
            var allRes: Array[(Int, Double)] = Array()
            try {
                val itemRowVector = itemPSMatrix.pull(itemId)
                val itemRowResult = itemRowVector.getStorage.asInstanceOf[DoubleVectorStorage].getValues
                val partitionKeys = PSAgentContext.get().getMatrixMetaManager().getPartitions(itemMatrixId)

                for (key <- partitionKeys) {
                    println(s"predictSingle_partitionKey=$key")
                    val numRows = key.getEndRow - key.getStartRow
                    val rowList = (key.getStartRow until key.getEndRow).toArray
                    try {
                        val vectors = itemPSMatrix.pull(rowList, numRows)
                        println("predictSingle_itemRow=", itemRowResult.mkString(", "))
                        val res = vectors.map(f => (f.getRowId, itemRowVector.dot(f) / (itemRowVector.norm() * f.norm()))).sortBy(_._2).reverse
                        println("predictSingle_partition_res=", res.slice(0, topK).mkString(", "))
                        allRes = allRes ++ res.slice(0, topK)
                    } catch {
                        case e: Throwable => println("predict Error!")
                    }

                }
                allRes = allRes.sortBy(_._2).reverse.slice(0, topK)
                println("predictSingle_all_res=", allRes.mkString(", "))
            }catch {
                case e:Throwable => println("predict error!, itemId=", itemId)
            }
            (itemId, allRes)
        }
        PSContext.instance()
        iter.zipWithIndex.map { case (batch, index) =>
            predictSingle(partitionId, batch, topK)
        }
    }

    // 模型训练,
    def train(trainSet: RDD[(Int, Array[Int], Array[Int])], numEpoch: Int,
              learningRate: Double, batchSize: Int, negative: Int,
              path: String): Unit = {
        //        val trainIter = NCFModel.buildDataBatches(trainSet, batchSize)
        for (epoch <- 1 to numEpoch) {
            val alpha = learningRate
            //            val data = trainIter.next()
            val data = trainSet
            println(s"data_epoch=$epoch, partition_count=", data.getNumPartitions)
            val numPartitions = data.getNumPartitions
            println("\nmodelTrain_numPartitions=", numPartitions)
            try {
                // cal batch loss and gradient
                val middle = data.mapPartitionsWithIndex((partitionId, iter) =>
                    sgdForPartition(partitionId, iter, numPartitions, negative, alpha, batchSize),
                    preservesPartitioning = true).collect()
                println("train_middle_value=", middle.mkString(", "))
                // cal batch total loss
                val loss = middle.map(f => f._1).sum
                println("train_epoch_" + epoch + "_loss=", loss)
                // model save
                if (epoch % 2 == 0 && epoch / 2 > 1) {
                    //                allPSMatrix.checkpoint(epoch)
                }
            } catch {
                case e: Throwable => println("epoch_", epoch, " error! ", e.printStackTrace())
            }
        }
    }

    def sgdForPartition(partitionId: Int, iter: Iterator[(Int, Array[Int], Array[Int])],
                        numPartitions: Int, negative: Int, alpha: Double, batchSize: Int):
    Iterator[(Double, Array[Array[Double]])] = {
        def sgdForSingleBatch(partitionId: Int, batch: Array[(Int, Array[Int], Array[Int])],
                              index: Int): (Double, Array[Array[Double]]) = {
            println("singleBatch_partitionID=", partitionId, " index=", index)
            try {
                val batchY = batch.map { case f =>
                    try {
                        val userVector = userPSMatrix.pull(f._1)
                        val dotRes = f._2.map{p =>
                            val pitemVector = itemPSMatrix.pull(p)
                            sigmoid(pitemVector.dot(userVector))
                        }
                        dotRes
                    } catch {
                        case e: Throwable => {
                            println("singleBatch_batchY_partitionID=", partitionId, " index=", index, " error! ", e.printStackTrace())
                            new Array[Double](f._2.length).map(f => 0.0)
                        }
                    }

                }
//                batchY.foreach(p => println("sgdForSingleBatch_modelRes=", p.mkString(", ")))
                // 计算梯度损失
                val loss = doBatchLoss(batchY)
                println("sgdForSingleBatch_mean_loss=", loss)
                //            batchY.foreach(p => println("sgdForSingleBatch_grad_loss=", p.mkString(", ")))
                // 梯度调整并回传
                adjustGrad(partitionId, batchY, batch, alpha, index)
                println("sgdForSingleBatch_finish_partitionId=", partitionId, " index=", index)
                (loss, batchY)
            } catch {
                case e: Throwable => {
                    println("singleBatch_partitionID=", partitionId, " index=", index, " error! ", e.printStackTrace())
                    (0, new Array[Array[Double]](0))
                }
            }
        }

        PSContext.instance()
        val batchIter = new Iterator[Array[(Int, Array[Int], Array[Int])]] {
            override def hasNext: Boolean = iter.hasNext

            override def next(): Array[(Int, Array[Int], Array[Int])] = {
                val batchData = new Array[(Int, Array[Int], Array[Int])](batchSize)
                var num = 0
                while (iter.hasNext && num < batchSize) {
                    batchData(num) = iter.next()
                    num += 1
                }
                if (num < batchSize) batchData.take(num) else batchData
            }
        }

        val res = batchIter.zipWithIndex.map { case (batch, index) =>
            sgdForSingleBatch(partitionId, batch, index)
        }
        println("sgdForPartition_finish_partitionId=", partitionId)
        res
    }

    // 梯度回传
    def adjustGrad(partitionId: Int, batchY: Array[Array[Double]],
                   batch: Array[(Int, Array[Int], Array[Int])],
                   alpha: Double, index: Int): Unit = {
        // 自己推导梯度公式
        //                f(i) = 2 * alpha * (label - y_^) * y_^ * (1 - y_^)
        batchY.zip(batch).foreach { case (y, signleData) =>
            val uid = signleData._1
            val itemIds = signleData._2
            val label = 1.0
            val item_grad = new mutable.HashMap[Int, Vector]()
            try {
                val userRowVector = userPSMatrix.pull(uid)
                val userRowValues = userRowVector.getStorage.asInstanceOf[DoubleVectorStorage].getValues
                // 计算物品Embedding梯度并更新,// 保存物品Embedding的均值数组，用于后面计算用户梯度
                val mean = new Array[Double](numCol)
                itemIds.zip(y).foreach { case (itemId, y_^) =>
                    try {
                        val itemRowVector = itemPSMatrix.pull(itemId)
                        val itemRowValues = itemRowVector.getStorage.asInstanceOf[DoubleVectorStorage].getValues
                        val l = -2 * y_^ * (1 - y_^) * (label - y_^)
                        val delta = userRowValues.map(f => f * l)

                        // 计算物品Embedding均值与物品梯度
                        for (i <- itemRowValues.indices) {
                            mean(i) += itemRowValues(i) * l
                            itemRowValues(i) = itemRowValues(i) + alpha * delta(i)
                        }
                        // 保存远程物品Embedding
//                        itemPSMatrix.update(itemId, itemRowVector)
                        item_grad.put(itemId, itemRowVector)
                    } catch {
                        case e: Throwable => println("adjustGrad_calGrad error! partitionID=", partitionId, " index=", index, e.printStackTrace())
                    }
                }
                // 计算用户Embedding梯度并更新
                val userDelta = mean.map(f => f / itemIds.length)
                for (i <- userRowValues.indices) {
                    userRowValues(i) = userRowValues(i) + alpha * userDelta(i)
                }
                userPSMatrix.update(uid, userRowVector)
                // 更新物品Embedding
                println("item_grad=", item_grad.mkString(","))
                item_grad.map(f=>itemPSMatrix.update(f._1, f._2))
            } catch {
                case e: Throwable => println("adjustGrad_error! partitionId=", partitionId, " index=", index, e.printStackTrace())
            }
        }
    }

    // 随机梯度下降，其他优化算法，后续再增加
    def doBatchLoss(batchY: Array[Array[Double]]): Double = {
        val loss = batchY.map { f =>
            var batchLoss = 0.0
            for (i <- f.indices) {
                //                val prob = FastSigmoid.sigmoid(dots(i)) // angel的快速计算库
                // 当前全为正样本计算,正样本标签概率为1，预测概率为prob
                // 计算方式性能有问题，后续再修改
                val label = 1.0
                // prob即y^
                val y_^ = f(i)
                // 总的loss，loss公式自己推导
                batchLoss -= Math.pow(label - y_^, 2)
            }
            batchLoss
        }
        // 返回当前batch的总loss
        loss.sum / batchY.map(p => p.length).sum
    }
    // 模型前向计算
    def forward(userVector: Vector, itemVector: Vector):Double={
        val gmf_val = userVector.dot(itemVector)
        val mlp1 = sigmoid(userVector.dot(w1Matrix.pull(0)) + itemVector.dot(w1Matrix.pull(1)) + bias1)
        val res_vec = VFactory.denseDoubleVector(Array(gmf_val, mlp1))
        val y = sigmoid(h1Vector.pull().dot(res_vec))
        y
    }
    // 模型反向计算梯度
    def backward(loss:Double, itemVector: Vector, userVector: Vector):
    (Vector, Vector, Vector, Double, Vector, Vector)={
        val g = itemVector.dot(userVector)
        val w1 = w1Matrix.pull(0)
        val w2 = w1Matrix.pull(1)
        val m = sigmoid(userVector.dot(w1) + itemVector.dot(w2) + bias1)
        val res_vec = VFactory.denseDoubleVector(Array(g, m))
        val hVec = h1Vector.pull()
        val hval = hVec.getStorage.asInstanceOf[DoubleVectorStorage].getValues
        val y = sigmoid(hVec.dot(res_vec))
        val m_d_b = m * (1-m)
        val m_d_user = w1.mul(m * (1-m))
        val m_d_item = w2.mul(m * (1-m))
        val g_d_user = itemVector
        val g_d_item = userVector

        val loss_d_y = 2 * (y-1)

        val y_d_h1 = y * (1-y) * g
        val y_d_h2 = y * (1-y) * m
        val y_d_m = y * (1-y) * hval(1)
        val y_d_g = y * (1-y) * hval(0)
        val y_d_b = y_d_m * m_d_b
        val y_d_w1 = userVector.mul(y_d_m * m * (1-m))
        val y_d_w2 = itemVector.mul(y_d_m * m * (1-m))
        val y_d_user = m_d_user.mul(y_d_m).add( g_d_user.mul(y_d_g))
        val y_d_item = m_d_item.mul(m_d_item).add(g_d_item.mul(y_d_g))

        (VFactory.denseDoubleVector(Array(loss_d_y * y_d_h1, loss_d_y * y_d_h2)),
            y_d_w1.mul(loss_d_y), y_d_w2.mul(loss_d_y), loss_d_y * y_d_b,
            y_d_user.mul(loss_d_y),
            y_d_item.mul(loss_d_y))
    }

    def sigmoid(x: Double): Double = 1.0 / (1.0 + Math.exp(-x))

    def log(x: Double): Double = Math.log(x)

    def save(modelPathRoot: String, epoch: Int): Unit = {
        println("start to save model")
        val modelPath = new Path(modelPathRoot, s"CP_$epoch").toString
        println("save_model_path=", modelPath)
        val sparkSession = SparkSession.builder().getOrCreate()
        println("finish_save_model")
    }

    // 正态分布初始化
    private def randomInitialize(matrix: PSMatrix, seed: Int): Unit = {
        // 用户和物品Embedding放在同一个矩阵中，低维表示用户，高维表示物品
        val partKey = PSAgentContext.get().getMatrixMetaManager.getPartitions(matrix.id)
        for (key <- partKey) {
            val numRows = key.getEndRow - key.getStartRow
            matrix.psfUpdate(new RandomNormal(matrix.id, key.getStartRow, numRows, 0.1, 1.0))
        }
    }

    private def createVector(numCol:Int):PSVector={
        val dense = PSVector.dense(10)
        VectorUtils.randomNormal(dense, 0.0, 1.0)
        return dense
    }
    // 创建矩阵
    private def createPSMatrix(numRow: Int, numCol: Int,
                               rowsInBlock: Int, colsInBlock: Int): PSMatrix = {
        val psMatrix = PSMatrix.dense(numRow, numCol, rowsInBlock, colsInBlock,
            RowType.T_DOUBLE_DENSE,
            Map(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS -> classOf[PartitionSourceArray].getName))
        psMatrix
    }
}

object NCFModel {
    def buildDataBatches(trainSet: RDD[(Int, Array[Int], Array[Int])],
                         batchSize: Int): Iterator[RDD[Array[(Int, Array[Int], Array[Int])]]] = {
        new Iterator[RDD[Array[(Int, Array[Int], Array[Int])]]] with Serializable {
            override def hasNext(): Boolean = true

            override def next(): RDD[Array[(Int, Array[Int], Array[Int])]] = {
                trainSet.mapPartitions { iter =>
                    val batchData = new Array[(Int, Array[Int], Array[Int])](batchSize)
                    new Iterator[Array[(Int, Array[Int], Array[Int])]] {
                        override def hasNext: Boolean = iter.hasNext

                        override def next(): Array[(Int, Array[Int], Array[Int])] = {
                            var num = 0
                            while (iter.hasNext && num < batchSize) {
                                batchData(num) = iter.next()
                                num += 1
                            }
                            if (num < batchSize) batchData.take(num) else batchData
                        }
                    }
                }
            }
        }
    }
}
