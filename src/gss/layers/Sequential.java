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
	public Data forward(Data input)
	{
		if (modules.size() == 0)
			return null;
		Data out=input;
		for (Module m:modules)
			out = m.forward(out);
		return out;
	}
	@Override
	public ArrayList<Data> getParameters()
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
