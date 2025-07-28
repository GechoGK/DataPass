import gss.arr.*;
import gss.layers.*;

import static gss.Util.*;

public class Main
{
	public static void main(String[] args)
	{

		new Main().a();

		System.out.println(line(54));

	}
	void a()
	{


	}
	void test5()
	{
		System.out.println(decString("Test 5.0 Conv1d layer test.", "=", 10));
		Data d1=NDArray.arange(12 * 2).reshapeLocal(2, 12);
		System.out.println(d1);
		d1.printArray();
		System.out.println(line(30));

		Conv1d c1=new Conv1d(12, 2, 5, 3);
		Data out=c1.forward(d1);
		System.out.println(out);
		out.printArray();


	}
	void test4()
	{
		System.out.println(decString("Test 4.0 MaxPool1d module test.", "=", 10));

		Data d1=NDArray.arange(5 * 12).reshapeLocal(5, -1);
		System.out.println(d1);
		d1.printArray();
		System.out.println(line(30));

		MaxPool1d m1=new MaxPool1d(3);
		Data d2=m1.forward(d1);
		System.out.println(d2);
		d2.printArray();

	}
	void test3()
	{
		System.out.println(decString("Test 3.0 Dropout module test.", "=", 10));
		Data d1=NDArray.rand(3, 5);

		Dropout d=new Dropout(0.5f);
		System.out.println(d1);
		d1.printArray();
		System.out.println(line(30));
		Data d2=d.forward(d1);
		System.out.println(d2);
		d2.printArray();

	}
	public static void test2()
	{
		System.out.println(decString("Test 2.0 Sequential module test.", "=", 10));
		Data d1=NDArray.arange(10).reshape(2, 5);
		System.out.println("input: " + d1);
		Sequential sq=new Sequential();
		sq.add(
			new Linear(5, 7, true),
			new Linear(7, 10, true),
			new Linear(10, 25, true),
			new Linear(25, 7, true));

		Data out=sq.forward(d1);
		System.out.println("output: " + out);

	}
	public static void test1()
	{
		System.out.println(decString("Test 1.0 Linear module test.", "=", 10));
		System.out.println("input");
		Data d1=NDArray.arange(10).reshape(2, 5);
		System.out.println(d1);

		Linear l1=new Linear(5, 10, false);
		Data out=l1.forward(d1);
		System.out.println("output");
		System.out.println(out);
		line(35);
		System.out.println("linear weights and biases");
		System.out.println(l1.weight);
		System.out.println(l1.biase);

	}
}
