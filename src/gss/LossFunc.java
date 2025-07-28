package gss;

import gss.arr.*;

public abstract class LossFunc extends Module
{
	public abstract Data forward(Data predicted, Data trueLabel);
	@Override
	public Data forward(Data arr)
	{
		throw new RuntimeException("this forward method is not implemented.");
	}
}
