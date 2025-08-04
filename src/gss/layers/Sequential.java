package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;

public class Sequential extends Module
{
	private List<Module> modules=new ArrayList<>();
	private boolean paramsCached=false;

	public Sequential(Module...mds)
	{
		for (Module m:mds)
			if (!modules.contains(m))
				modules.add(m);
	}
	public Module add(Module...mds)
	{
		for (Module m:mds)
			modules.add(m);
		paramsCached = false;
		return this;
	}
	@Override
	public Base forward(Base input)
	{
		if (modules.size() == 0)
			return null;
		Base out=input;
		for (Module m:modules)
		{
			// System.out.print(".");
			out = m.forward(out);
			// System.out.println(out);
		}
		// System.out.println();
		return out;
	}
	@Override
	public ArrayList<Base> getParameters()
	{
		if (!paramsCached)
		{
			params.clear();
			for (Module m:modules)
				params.addAll(m.getParameters());
			paramsCached = true;
		}
		return params;
	}
}
