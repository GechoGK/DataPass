package modules;

import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;

import static gss.Util.*;

public class XOR extends Sequential
{
	public XOR()
	{
		init();
	}
	public void init()
	{
		int hiddenSize=5;
		// it works with hidden size starts from 2 upto ...

		Tanh tan=new Tanh();
		add(new Linear(2, hiddenSize), tan, new Linear(hiddenSize, 1), tan);

	}
}
