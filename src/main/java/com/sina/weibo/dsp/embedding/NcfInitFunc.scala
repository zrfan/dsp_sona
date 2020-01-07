package com.sina.weibo.dsp.embedding

import com.sina.weibo.dsp.embedding.NcfInitFunc.{NcfInitParam, NcfPartitionInitParam}
import com.tencent.angel.PartitionKey
import com.tencent.angel.ml.math2.storage.IntFloatDenseVectorStorage
import com.tencent.angel.ml.matrix.psf.update.base.{PartitionUpdateParam, UpdateFunc, UpdateParam}
import com.tencent.angel.ps.storage.partition.RowBasedPartition
import com.tencent.angel.ps.storage.vector.ServerRowUtils
import com.tencent.angel.psagent.PSAgentContext
import io.netty.buffer.ByteBuf

import scala.util.Random
import scala.collection.JavaConversions._

class NcfInitFunc(param: NcfInitParam) extends UpdateFunc(param) {
    def this(matrixId: Int) = this(new NcfInitParam(matrixId))
    def this() = this(null)

    override def partitionUpdate(partitionUpdateParam: PartitionUpdateParam): Unit = {
        println("start_to_update_matrix_param=", partitionUpdateParam.toString)
        val part = psContext.getMatrixStorageManager.getPart(partitionUpdateParam.getMatrixId,
            partitionUpdateParam.getPartKey.getPartitionId)
            .asInstanceOf[RowBasedPartition]
        println("getmatrix_storagemanager=", part.toString)
        if (part != null) {
            val ncfParam = partitionUpdateParam.asInstanceOf[NcfPartitionInitParam]
            println("updatematrix_param=", ncfParam.toString)
            update(part, partitionUpdateParam.getPartKey)
        }
    }

    private def update(partition: RowBasedPartition, key: PartitionKey): Unit = {
        val startRow = key.getStartRow
        val endRow = key.getEndRow
        val rand = 10
        println("update_startRow=", startRow, " endRow=", endRow, " rand=", rand)
        (startRow until endRow).map(rowId => (rowId, rand)).par.foreach {
            case (rowId, rowSeed) =>
                val rowRandom = new Random(rowSeed)
                val data = ServerRowUtils.getVector(partition.getRow(rowId))
                    .getStorage.asInstanceOf[IntFloatDenseVectorStorage].getValues
                println("duringInitialize_datavalue=", data.toString)
                data.indices.foreach(data(_) = (rowRandom.nextFloat() - 0.5f))
                println("datas=", data.toString)
        }
    }
}

object NcfInitFunc {

    class NcfPartitionInitParam(matrixId: Int, partKey: PartitionKey)
        extends PartitionUpdateParam(matrixId, partKey) {
        def this() = this(-1, null)

        override def serialize(buf: ByteBuf): Unit = {
            super.serialize(buf)
            //            buf.writeInt(seed)
        }

        override def deserialize(buf: ByteBuf): Unit = {
            println("\nNcfInitFunc_deserialize call")
            super.deserialize(buf)
            //            this.seed = buf.readInt()
        }

        override def bufferLen(): Int = super.bufferLen
    }

    class NcfInitParam(matrixId: Int) extends UpdateParam(matrixId) {
        override def split: java.util.List[PartitionUpdateParam] = {
            val pkey = PSAgentContext.get.getMatrixMetaManager.getPartitions(matrixId)
            println("\nNCFInitParam_split initializeMatrix_pkey=", pkey.toString, " length=", pkey.length)
            val k0 = pkey(0)
            println("\nNCFInitParam_split pkey0=", k0.toString)
            val p = new NcfPartitionInitParam(matrixId, k0)
            println("\nNCFInitParam_split ncfpartitionInitParam=", p.toString)

            println("\nNCFInitParam_split map_result=", pkey.map(part => new NcfPartitionInitParam(matrixId, part)).mkString("\n"))
            PSAgentContext.get.getMatrixMetaManager.getPartitions(matrixId).map { part =>
                new NcfPartitionInitParam(matrixId, part)
            }
        }
    }

}
