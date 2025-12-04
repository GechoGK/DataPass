package gss;

import gss.arr.*;
import java.util.*;

public abstract class Module
{
	public ArrayList<Base> params=new ArrayList<>();

	public abstract Base forward(Base dataIn);

	public Base newParam(Base arr)
	{
		if (!arr.hasGradient())
			arr.setRequiresGradient(true);
		if (!params.contains(arr))
			params.add(arr);
		return arr;
	}
	public ArrayList<Base> getParameters()
	{
		return params;
	}
}
