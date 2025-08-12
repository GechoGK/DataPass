import gss.*;
import gss.act.*;
import gss.arr.*;

import static gss.Util.*;
import static gss.arr.NDArray.*;

public class Main
{
	public static void main(String[] args)
	{

		new Main().a();

	}
	void a()
	{

		Base b1=NDArray.arange(30).reshapeLocal(3, 10).setRequiresGradient(true);
		Base k1=NDArray.arange(5).reshapeLocal(5).setRequiresGradient(true);

		Base o1=NDArray.convolve1d(b1, k1);
		b1.printArray();
		print(line(7));
		k1.printArray();
		print(line(10));
		o1.printArray();
		print(line(30));
		o1.setGrad(NDArray.arange(1, o1.length + 1));
		o1.backward();
		b1.detachGradient().printArray();
		print(line(10));
		k1.detachGradient().printArray();
		print(line(30));
	}
	void test4()
	{
		// not tested with value type.
		// dot2 doesn't use Value class, so use Value class instead.
		print(decString("Test 4.0 dot product", 10));

		Base b1=NDArray.arange(20).reshapeLocal(2, 2, 5).setRequiresGradient(true);
		Base b2=NDArray.arange(50).reshapeLocal(2, 5, 5).setRequiresGradient(true);

		Base b3=NDArray.dot(b1, b2);
		b1.printArray();
		print(line(10));
		b2.printArray();
		print(b3);
		b3.printArray();
		print(line(20));
		b3.setGrad(NDArray.arange(1, 41));
		b3.backward();
		b1.detachGradient().printArray();
		print(line(5));
		b2.detachGradient().printArray();
		print(line(30));

		Base d1=NDArray.arange(20).reshapeLocal(2, 2, 5).setRequiresGradient(true);
		Base d2=NDArray.arange(50).reshapeLocal(2, 5, 5).setRequiresGradient(true);

		Base d3=TestValue.dot2(d1, d2);
		d1.printArray();
		print(line(10));
		d2.printArray();
		print(d3);
		d3.printArray();
		print(line(20));
		d3.setGrad(NDArray.arange(1, 41));
		d3.backward();
		d1.detachGradient().printArray();
		print(line(5));
		d2.detachGradient().printArray();
		print(line(30));

		if (Util.equals(b1, d1) && Util.equals(b2, d2))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array data equality.");
		if (Util.equals(b1.detachGradient(), d1.detachGradient()) && Util.equals(b2.detachGradient(), d2.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array gradient equality.");
	}
	void test3()
	{
		print(decString("Test 3.0 sigmoid test", 10));
		Base bg=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base o=new Tanh().forward(bg);
		o.printArray();
		o.setGrad(NDArray.arange(1, 21));
		o.backward();
		print(line(10));
		bg.detachGradient().printArray();
		print(decString("compare", 7));

		Base b=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base b2=TestValue.tanh(b);
		print(b2);
		b2.printArray();
		print(line(10));
		b2.setGrad(NDArray.arange(1, 21));
		b2.backward();
		b.detachGradient().printArray();

		if (Util.equals(bg, b))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
		if (Util.equals(bg.detachGradient(), b.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
	}
	void test2()
	{
		// sigmoid gradient have problem.
		print(decString("Test 2.0 sigmoid test", 10));
		Base bg=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base o=new Sigmoid().forward(bg);
		o.printArray();
		o.setGrad(NDArray.arange(20));
		o.backward();
		print(line(10));
		bg.detachGradient().printArray();
		print(decString("compare", 7));

		Base b=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base b2=TestValue.sigmoid(b);
		print(b2);
		b2.printArray();
		print(line(10));
		b2.setGrad(NDArray.arange(20));
		b2.backward();
		b.detachGradient().printArray();

		if (Util.equals(bg, b))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
		if (Util.equals(bg.detachGradient(), b.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
	}
	void test1()
	{
		print(decString("Test 1.0 Relu test", 10));

		Base bg=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base o=new Relu().forward(bg);
		o.printArray();
		o.setGrad(NDArray.arange(1, 21));
		o.backward();
		print(line(10));
		bg.detachGradient().printArray();
		print(decString("compare", 7));

		Base b=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base b2=TestValue.relu(b);
		print(b2);
		print(line(10));
		b2.setGrad(NDArray.arange(1, 21));
		b2.backward();
		b.detachGradient().printArray();

		if (Util.equals(bg, b))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
		if (Util.equals(bg.detachGradient(), b.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
	}
}
