package gss.layers;

import gss.*;
import gss.arr.*;

import static gss.Util.*;

public class LSTM extends Module
{
	private int inputSize,hiddenSize;
	private boolean hasBiase;

	public Base forget_ih_weight,input_ih_wight,cand_ih_wight,output_ih_wight;
	public Base forget_hh_weight,input_hh_wight,cand_hh_wight,output_hh_wight;
	public Base forget_biase,input_biase,cand_biase,output_biase;

	public Base hiddenState,cellState;

	public Base W;

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
		forget_ih_weight = newParam(NDArray.rand(inputSize, hiddenSize));
		input_ih_wight = newParam(NDArray.rand(inputSize, hiddenSize));
		cand_ih_wight = newParam(NDArray.rand(inputSize, hiddenSize));
		output_ih_wight = newParam(NDArray.rand(inputSize, hiddenSize));

		forget_hh_weight = newParam(NDArray.rand(hiddenSize, hiddenSize));
		input_hh_wight = newParam(NDArray.rand(hiddenSize, hiddenSize));
		cand_hh_wight = newParam(NDArray.rand(hiddenSize, hiddenSize));
		output_hh_wight = newParam(NDArray.rand(hiddenSize, hiddenSize));

		forget_biase = newParam(NDArray.ones(hiddenSize));
		input_biase = newParam(NDArray.ones(hiddenSize));
		cand_biase = newParam(NDArray.ones(hiddenSize));
		output_biase = newParam(NDArray.ones(hiddenSize));

		hiddenState = NDArray.empty(hiddenSize);
		cellState = NDArray.empty(hiddenSize);


		W = newParam(NDArray.rand(4 * hiddenSize, hiddenSize + inputSize));

	}
	@Override
	public Base forward(Base dataIn)
	{
		Base comb=NDArray.concat(dataIn, hiddenState, 0);
		println(dataIn, hiddenState, comb);
		return null;
	}
}
