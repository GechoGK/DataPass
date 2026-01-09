package gss.layers;

import gss.*;
import gss.arr.*;

import static gss.Util.*;

public class Embedding extends Module
{
	public int vocabSize,embedding_dim;
	// private Base embeddingWeight;

	/*
	 it needs custom implementation of dot product.
	 the default dot proruct took too long to complete on large datasets.
	 infact most of the computation are useless.
	 */

	public Linear layer;

	public Embedding(int vocab_size, int embed_dim)
	{
		this.vocabSize = vocab_size;
		this.embedding_dim = embed_dim;
		init();
	}
	private void init()
	{
		// embeddingWeight = newParam(NDArray.rand(vocabSize, embedding_dim));
		// we cam handle without linear layer, it is because onehot verctors have so many 0 values, so it is waste of time.
		layer = new Linear(vocabSize, embedding_dim, false);
	}
	@Override
	public Base forward(Base dataIn)
	{
		// custom forward or custom dot product for onehot vector.
		return layer.forward(dataIn);
	}
	public Base forwardWithIndices(Base indicesData)
	{
		// direct index access.
		Base out=NDArray.empty(append(indicesData.shape, embedding_dim));
		Base outD=out.as2DArray();
		for (int i=0;i < indicesData.length;i++)
		{
			outD.slice(i).set(layer.weight.slice((int)indicesData.get(i)));
		}
		out.setRequiresGradient(indicesData.hasGradient() || layer.weight.hasGradient());
		out.setGradientFunctionS(GradFunc.indexGradient, indicesData, layer.weight);
		return out;
	}
}
