package gss.layers;

import gss.*;
import gss.arr.*;

public class Flatten extends Module
{
	@Override
	public Base forward(Base dataIn)
	{
		return dataIn.reshape(-1);
	}
}
