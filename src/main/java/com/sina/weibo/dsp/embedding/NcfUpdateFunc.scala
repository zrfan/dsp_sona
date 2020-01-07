package com.sina.weibo.dsp.embedding

import com.tencent.angel.ml.matrix.psf.update.enhance.{MultiRowUpdateFunc, MultiRowUpdateParam}
import com.tencent.angel.ps.storage.vector.ServerRow

class NcfUpdateFunc(param: MultiRowUpdateParam) extends MultiRowUpdateFunc{
    def this(matrixId: Int, rowIds: Array[Int], values: Array[Array[Double]]) = {
        this(new MultiRowUpdateParam(matrixId, rowIds, values))
    }
    override def update(row: ServerRow, values: Array[Double]): Unit = {
        println("ncfUpdateFunc")
    }
}
