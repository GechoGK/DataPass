package gss;

import gss.arr.*;
import java.util.*;

public abstract class Module
{
	public ArrayList<Base> params=new ArrayList<>();
	public ArrayList<Module> subModules=new ArrayList<>();

	public abstract Base forward(Base dataIn);

	public Base newParam(Base prm)
	{
		// repeating gradient is not allowed.
		if (!prm.hasGradient())
			prm.setRequiresGradient(true);
		if (!params.contains(prm))
			params.add(prm);
		return prm;
	}
	public Module newSubModule(Module m)
	{
		// repeating module is allowed. to forward one module multiple times,
		// but repeating parameter would cause problem on gradient calculations.
		// if (!subModules.contains(m))
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
