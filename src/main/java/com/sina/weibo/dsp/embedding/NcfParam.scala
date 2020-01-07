package com.sina.weibo.dsp.embedding

class NcfParam extends Serializable {
    var partitionNum: Int = _
    var embeddingDim: Int = _
    var negSample: Int = _
    var numEpoch: Int = _
    var batchSize: Int = _
    var maxIndex: Int = _
    var seed: Int = _
    var modelPath: String = _
    var numRow: Int = _
    var numCol: Int = _
    var rowsInBlock: Int = _
    var colsInBlock: Int = _
    def setPartitionNum(partitionNum: Int): this.type ={
        this.partitionNum = partitionNum
        this
    }
    def setEmbeddingDim(embeddingDim: Int): this.type ={
        this.embeddingDim = embeddingDim
        this
    }
    def setNegSample(negSample: Int): this.type ={
        this.negSample = negSample
        this
    }
    def setNumEpoch(numEpoch: Int): this.type ={
        this.numEpoch = numEpoch
        this
    }
    def setBatchSize(batchSize: Int): this.type ={
        this.batchSize = batchSize
        this
    }
    def setMaxIndex(maxIndex: Int): this.type ={
        this.maxIndex = maxIndex
        this
    }
    def setSeed(seed: Int):this.type = {
        this.seed = seed
        this
    }
    def setModelPath(modelPath: String):this.type ={
        this.modelPath = modelPath
        this
    }
    def setNumRow(numRow: Int):this.type = {
        this.numRow = numRow
        this
    }
    def setNumCol(numCol: Int):this.type = {
        this.numCol = numCol
        this
    }
    def setRowsInBlock(rowsInBlock: Int):this.type = {
        this.rowsInBlock = rowsInBlock
        this
    }
    def setColsInBlock(colsInBlock: Int):this.type = {
        this.colsInBlock = colsInBlock
        this
    }

}
