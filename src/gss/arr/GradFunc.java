package gss.arr;

public abstract class GradFunc
{
	// name for debugging purpose.
	private String name;

	public GradFunc()
	{
		this.name = "unknown";
	}
	public GradFunc(String name)
	{
		this.name = name;
	}
	public abstract Data backward(Data host, Data...childs, Object params)
	@Override
	public String toString()
	{
		return name + "Gradient[" + hashCode() + "]";
	}
}
