package modules;

import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;

import static gss.Util.*;

public class XOR extends Sequential
{
	private Module l1,a1,l2,a2;

	public XOR()
	{
		init();
	}
	public void init()
	{
		int hiddenSize=5;
		// it works with hidden size starts from 2 upto ...

		l1 = add(new Linear(2, hiddenSize));
		a1 = add(new Tanh());
		l2 = add(new Linear(hiddenSize, 1));
		a2 = add(new Tanh());

		
	}
	// @Override
	public Base forward2(Base dataIn)
	{
		Base X = l1.forward(dataIn);
		X = a1.forward(X);
		X = l2.forward(X);
		X = a2.forward(X);
		return X;
	}
}
