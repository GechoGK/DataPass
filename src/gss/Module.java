package gss;

import gss.arr.*;
import java.util.*;

public abstract class Module
{
	public ArrayList<Data> params=new ArrayList<>();

	public abstract Data forward(Data dataIn);

	public Data newParam(Data arr)
	{
		if (!params.contains(arr))
			params.add(arr);
		return arr;
	}
	public ArrayList<Data> getParameters()
	{
		return params;
	}
}
