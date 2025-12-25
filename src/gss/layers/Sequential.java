package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;

public class Sequential extends Module
{
	public Sequential(Module...mds)
	{
		add(mds);
	}
	public Module add(Module...mds)
	{
		for (Module m:mds)
			newSubModule(m);
		return this;
	}
	@Override
	public Base forward(Base input)
	{
		if (subModules.size() == 0)
			return null;
		Base out=input;
		for (Module m:subModules)
		{
			// System.out.print(".");
			out = m.forward(out);
			// System.out.println(out);
		}
		// System.out.println();
		return out;
	}
}
