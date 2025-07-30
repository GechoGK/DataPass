package gss;

import gss.arr.*;

public abstract class LossFunc extends Module
{
	public abstract Base forward(Base predicted, Base trueLabel);
	@Override
	public Base forward(Base arr)
	{
		throw new RuntimeException("this forward method is not implemented.");
	}
}
