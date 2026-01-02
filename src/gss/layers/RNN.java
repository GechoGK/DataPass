package gss.layers;

import gss.*;
import gss.act.*;
import gss.arr.*;

public class RNN extends Module
{
	private int input;
	private int hidden;
	private boolean hasBiase;
	// remember batch first is false;
	// the data type should be (sequence_length, batch_size, input_length);

	public Base ih_weight,hh_weight,biase;

	public RNN(int in, int hiddenOrOutput)
	{
		this(in, hiddenOrOutput, true);
	}
	public RNN(int in, int hiddenOrOutput, boolean hasBiase)
	{
		this.input = in;
		this.hidden = hiddenOrOutput;
		this.hasBiase = hasBiase;
		init();
	}
	private void init()
	{
		// input to hidden weight;
		ih_weight = newParam(NDArray.rand(input, hidden));
		// hidden to hidden weight.
		hh_weight = newParam(NDArray.zeros(hidden, hidden));
		// biase if enabled.
		if (hasBiase)
			biase = newParam(NDArray.ones(hidden));
		// hh_biase = newParam(NDArray.ones(hidden));
	}
	@Override
	public Base forward(Base dataIn)
	{
		// dataIn should be (sequence_len, batch_size, input);
		// modify the inout to 3d array(sequence_length, batch_size, input_size);
		dataIn = dataIn.as3DArray();
		Base hidden_state=NDArray.zeros(hidden);
		for (int i=0;i < dataIn.shape[0];i++)
		{
			// in_w . hh_w + biase;
			// print("sequence", i);
			// (input_to_hidden . hidden_to_hidden);
			Base ih_rs=NDArray.dot(dataIn.slice(i), ih_weight);
			// (hidden_to_hidden . hidden_to_hidden);
			Base hh_rs= NDArray.dot(hidden_state, hh_weight);
			// tanh( input_to_hidden_result + hidden_to_hidden_result + biase);
			Base hidden=NDArray.add(ih_rs, hh_rs);
			if (hasBiase)
				hidden = NDArray.add(hidden, biase);
			hidden_state = new Tanh().forward(hidden);
		}
		// output shape (batch_size, hidden_size)
		return hidden_state;
	}
}
