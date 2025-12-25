package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;

public class Sequential extends Module
{
	private List<Module> modules=new ArrayList<>();

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
}
