package gss.layers;

import gss.*;
import gss.act.*;
import gss.arr.*;

import static gss.Util.*;

public class LSTM extends Module
{
	private int inputSize,hiddenSize;
	private boolean hasBiase;

	public Base hiddenState,cellState;

	public Base W,B;

	public LSTM(int input_size, int hiddenSize)
	{
		this(input_size, hiddenSize, true);
	}
	public LSTM(int input_size, int hidden_size, boolean has_biase)
	{
		this.inputSize = input_size;
		this.hiddenSize = hidden_size;
		this.hasBiase = has_biase;
		init();
	}
	private  void init()
	{
		// example hidden = 4, input = 3;
		// W = newParam(NDArray.rand(4 * hiddenSize, hiddenSize + inputSize)); // 16,4+3 = (16,7)
		B = newParam(NDArray.ones(4 * hiddenSize)); // 16

		// the original weight is the above one.
		// we can swap the dimension of the weights in order to save some computation
		// like (W.transpose()) this prevent us from accesing as 1d aray.
		W = newParam(NDArray.rand(hiddenSize + inputSize, 4 * hiddenSize));
	}
	@Override
	public Base forward(Base dataIn)
	{
		// dataIn shape = (sequence_length, batch_size, input_length);
		// example hidden = 4, input = 3;
		dataIn = dataIn.as3DArray(); // convert into (sequence_length, batch_size, input_length)
		int sequence=dataIn.shape[0];
		int batch=dataIn.shape[1];
		hiddenState = newParam(NDArray.empty(batch, hiddenSize)); // (batch, hidden)
		cellState = NDArray.empty(batch, hiddenSize); // (batch, hidden)
		for (int i=0;i < sequence;i++)
		{
			// concat doesn't have gradientFunction.
			Base comb=NDArray.concat(hiddenState, dataIn.slice(i), 1); // 7 
			Base gate = NDArray.add(NDArray.dot(comb, W), B);  // 16
			Base fg=new Sigmoid().forward(gate.slice(new int[][]{r(-1),r(4)})); // 4
			Base ig=new Sigmoid().forward(gate.slice(new int[][]{r(-1),r(4, 8)})); // 4
			Base gg=new Tanh().forward(gate.slice(new int[][]{r(-1),r(8, 12)})); // 4
			Base og=new Sigmoid().forward(gate.slice(new int[][]{r(-1),r(12, 16)})); // 4

			cellState = NDArray.add(NDArray.mul(cellState, fg), NDArray.mul(ig, gg)); // 4

			hiddenState = NDArray.mul(og, new Tanh().forward(cellState)); // 4
		}
		return hiddenState;
	}
}
