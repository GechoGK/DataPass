package gss;

public class Functions
{
	public abstract static class MapFunction
	{
		public abstract float apply(float p1);
	}
	public abstract static class ZipFunction
	{
		public abstract float apply(float p1, float p2);
	}
	public abstract static class ArrayToFloatFunction
	{
		public abstract float apply(int[] p1);
	}
	public abstract static class MapConsumer
	{
		public abstract void consume(float p1);
	}
	public abstract static class ZipConsumer
	{
		public abstract void consume(float p1, float p2);
	}
	public abstract static class ArrayConsumer
	{
		public abstract void consume(int[] p1);
	}
}
