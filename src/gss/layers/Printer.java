package gss.layers;

import gss.*;
import gss.arr.*;

public class Printer extends Module
{
	@Override
	public Base forward(Base dataIn)
	{
		Util.print(Util.getString("--", 10));
		Util.println(dataIn);
		return dataIn;
	}
}
