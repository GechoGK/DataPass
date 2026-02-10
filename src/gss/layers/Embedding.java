package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class Embedding extends Module
{
	public int vocabSize,embedding_dim;
	public Base embeddingWeight;

	public Embedding(int vocab_size, int embed_dim)
	{
		this.vocabSize = vocab_size;
		this.embedding_dim = embed_dim;
		init();
	}
	private void init()
	{
		embeddingWeight = newParam(NDArray.rand(vocabSize, embedding_dim));
	}
	@Override
	public Base forward(Base dataIn)
	{
		// needs to support batching.
		// dataIn should be 2d array(batch, one hot vector);.
		Base in=dataIn.reshape(-1, vocabSize); // there is no need to cache to float array.
		if (in.shape[1] > vocabSize)
			error("one hot vector should not be greater than vocabulary size(" + in.length + " ≠ " + vocabSize + ")");
		ArrayList<float[]> ws=new ArrayList<>();
		ArrayList<Integer> ind=new ArrayList<>();
		for (int b=0;b < in.shape[0];b++)
			for (int i=0;i < in.shape[1];i++)
			{
				if (in.get(b, i) != 0) // grab the weight at "i" position.
				{
					ind.add(i);
					float[] f=new float[embedding_dim];
					for (int j=0;j < embedding_dim;j++)
					{
						f[j] = embeddingWeight.get(i, j);
					}
					ws.add(f);
				}
			}
		// now every item at index[dataIn] is collected.
		float[]o=new float[ws.size() * embedding_dim];
		int p=0;
		for (float[] ed:ws)
			for (float f:ed)
				o[p++] = f;
		Integer[]indices=ind.toArray(new Integer[0]);
		int[]shape={ws.size(),embedding_dim};
		Base out=NDArray.wrap(o, shape);
		out.setRequiresGradient(dataIn.hasGradient() | embeddingWeight.hasGradient());
		out.setGradientFunctionS(embedGradient, indices, embeddingWeight);
		ws.clear();
		return out;
	}
	// not tested. becarefull.
	public Base forwardWithIndices(Base indicesData)
	{
		// direct index access.
		float[]out=new float[indicesData.length * embedding_dim];
		Integer[]indices=new Integer[indicesData.length];;
		int pos=0;
		for (int i=0;i < indicesData.length;i++)
		{
			int ind=(int)indicesData.get1d(i);
			indices[i] = ind;
			for (int e=0;e < embedding_dim;e++)
			{
				out[pos++] = embeddingWeight.get(ind, e);
			}
		}
		Base o=NDArray.wrap(out, indicesData.length, embedding_dim);
		o.setRequiresGradient(indicesData.hasGradient() || embeddingWeight.hasGradient());
		o.setGradientFunctionS(embedGradient, indices, embeddingWeight);
		return o;
	}
	public static GradFunc embedGradient=new GradFunc("embedding"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Integer[] ind=(Integer[])params;
			Base a=childs[0];
			for (int i=0;i < ind.length;i++)
			{
				for (int j=0;j < host.shape[1];j++)
				{
					a.setGrad(ar(ind[i], j), host.getGrad(i, j));
				}
			}
			return null;
		}
	};
}
