package modules;

import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;

import static gss.Util.*;

public class XOR extends Sequential
{
	// it works with hidden size starts from 2 upto ...
	public int hiddenSize=5;

	public XOR()
	{
		init();
	}
	public XOR(int hiddenSize)
	{
		this.hiddenSize = hiddenSize;
	}
	public void init()
	{
		Activation sig=new Sigmoid();
		add(new Linear(2, hiddenSize),
			sig,
			new Linear(hiddenSize, 1),
			sig);

	}
}
