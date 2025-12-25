package gss;

import gss.arr.*;
import java.util.*;

public abstract class Module
{
	public ArrayList<Base> params=new ArrayList<>();
	public ArrayList<Module> subModules=new ArrayList<>();

	public abstract Base forward(Base dataIn);

	public Base newParam(Base arr)
	{
		if (!arr.hasGradient())
			arr.setRequiresGradient(true);
		if (!params.contains(arr))
			params.add(arr);
		return arr;
	}
	public Module newSubModule(Module m)
	{
		if (!subModules.contains(m))
			subModules.add(m);
		for (Base prm:m.params)
			if (!params.contains(prm))
				params.add(prm);
		return m;
	}
	public ArrayList<Base> getParameters()
	{
		return params;
	}
}
