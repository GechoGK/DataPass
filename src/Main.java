import gss.act.*;
import gss.arr.*;
import gss.layers.*;
import gss.lossfunctions.*;

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
	void test8()
	{
		System.out.println(decString("Test 8.0 Gradient function test.", "=", 10));
		System.out.println("=== tested on additionGradientTest");
		Data d1=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);
		Data d2=NDArray.arange(10).setRequiresGradient(true);

		Data d3=NDArray.add(d1, d2);

		System.out.println(d3);
		System.out.println(d3.gradientFunction);
		d3.printArray();
		System.out.println(line(30));

		d3.fillGrad(3);
		d3.backward();

		d1.detachGradient().printArray();
		System.out.println(line(10));
		d2.detachGradient().printArray();
		d2.zeroGrad();
		System.out.println("after zero gradient");
		d2.detachGradient().printArray();
	}
	void test7()
	{
		System.out.println(decString("Test 7.0 loss functions test.", "=", 10));
		Data pred=NDArray.rand(10);
		Data trueLabel=NDArray.zeros(10);
		trueLabel.set(new int[]{3}, 1);
		System.out.println("predicted :" + pred);
		pred.printArray();
		System.out.println(line(10));
		System.out.println("true label :" + trueLabel);
		trueLabel.printArray();
		System.out.println(line(30));

		System.out.println(decString("BCE",  10));
		Data out=new BCE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("MAE",  10));
		out = new MAE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("MCCE",  10));
		out = new MCCE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("MSE",  10));
		out = new MSE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("Multi label binary cross entropy",  10));
		out = new MultiLabelBinaryCrossEntropy().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

	}
	void test6()
	{
		System.out.println(decString("Test 6.0 activation layers test.", "=", 10));
		Data in=NDArray.arange(-5, 5).reshape(2, 5);
		System.out.println(in);
		in.printArray();
		System.out.println(line(30));

		System.out.println(decString("sigmoid activation", 10));
		Data out=new Sigmoid().forward(in);
		System.out.println(out);
		out.printArray();
		System.out.println(line(30) + "\n");

		System.out.println(decString("tanh activation", 10));
		out = new Tanh().forward(in);
		System.out.println(out);
		out.printArray();
		System.out.println(line(30) + "\n");

		System.out.println(decString("relu activation", 10));
		out = new Relu().forward(in);
		System.out.println(out);
		out.printArray();
		System.out.println(line(30) + "\n");

		System.out.println(decString("softmax activation", 10));
		out = new Softmax().forward(in);
		System.out.println(out);
		out.printArray();


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
