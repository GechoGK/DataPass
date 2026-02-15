package gss.layers;

import gss.*;
import gss.arr.*;

public class Flatten extends Module
{
	private boolean haveBatch;
	public Flatten()
	{}
	public Flatten(boolean haveBatch)
	{
		this.haveBatch = haveBatch;
	}
	@Override
	public Base forward(Base dataIn)
	{
		Base b=null;
		if (haveBatch)
			b = dataIn.reshape(dataIn.shape[0], -1);
		else
			b = dataIn.reshape(-1);
		// Util.println(b.shape);
		return b;
	}
}
