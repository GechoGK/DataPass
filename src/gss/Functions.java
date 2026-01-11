package gss;

public class Functions
{
	// returns a new modified float value 
	public abstract static class MapFunction
	{
		public abstract float apply(float p1);
	}
	// return a new float value, by using two float values.
	public abstract static class ZipFunction
	{
		public abstract float apply(float p1, float p2);
	}
	// return a new float value, by using shape of an array.
	public abstract static class ArrayFunction
	{
		public abstract float apply(int[] p1);
	}
	// use float value, doesn't return anything.
	public abstract static class MapConsumer
	{
		public abstract void consume(float p1);
	}
	// use two float values, doesn't return anything.
	public abstract static class ZipConsumer
	{
		public abstract void consume(float p1, float p2);
	}
	// use int array or shape array, doesn't return anything.
	public abstract static class ArrayConsumer
	{
		public abstract void consume(int[] p1);
	}
}
