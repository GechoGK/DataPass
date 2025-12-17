import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class Test1_NDArrayData
{
	public static void test()
	{

		test1();
		test2();
		test3();
		test4();
		test5();
		test6();
		test7();
		test8();
		test9();
		test10();
		test11();

	}
	public static void test11()
	{
		System.out.println("========== Test 11.0 convolution 1d and correlation 1d test. ==========");
		Base d1=NDArray.arange(40).reshape(-1, 20);
		Base k=new Base(new float[]{1,2,3,4,5});
		System.out.println(d1);
		d1.printArray();
		System.out.println(getString("-", 30));
		System.out.println(k);
		k.printArray();

		System.out.println("\n" + decString("convolution 1d", 30));
		Base d3=NDArray.convolve1d(d1.reshape(-1), k);
		System.out.println(d3);
		d3.printArray();

		System.out.println("\n" + decString("correlation 1d", 30));
		d3 = NDArray.correlate1d(d1.reshape(-1), k);
		System.out.println(d3);
		d3.printArray();

	}
	public static void test10()
	{
		print(decString("Test 8.0 dot product test.", 7));
		/*
		 // a = [[1,2,3],
		 // 	 [3,4,5]]	 
		 // b = [1,2,3]

		 c =  a.b = (2,3).(3)->(3,1)  3==3
		 c.shape = (2,1)-> (2)

		 */

		Base a=NDArray.arange(2 * 5).reshapeLocal(2, 5).setRequiresGradient(true);
		Base b=NDArray.arange(5 * 4).reshapeLocal(5, 4).setRequiresGradient(true);

		print(a);
		a.printArray();
		print(line(30));
		print(b);
		b.printArray();
		print(line(50));

		Base c=NDArray.dot(a, b);
		print(c);

		c.printArray();
		print(line(30));
		draw(c, "");

	}
	public static void draw(Base bs, String s)
	{
		System.out.println(s + bs.gradientFunction);
		if (bs.childs != null)
			for (Base b:bs.childs)
				draw(b, "   " + s);
	}
	public static void  test9()
	{
		System.out.println("========== Test 9.0 1d, 2d and 3d array options. ==========");

		Base d=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		d.printArray();

		System.out.println(getString("-", 20));
		System.out.println("1d array");
		Base d1 = d.as1DArray();
		System.out.println(d1);
		d1.printArray();
		System.out.println(getString("-", 20));

		System.out.println("2d array");
		d1 = d.as2DArray();
		System.out.println(d1);
		d1.printArray();
		System.out.println(getString("-", 20));

		System.out.println("3d array");
		d1 = d.as3DArray();
		System.out.println(d1);
		d1.printArray();
		System.out.println(getString("-", 20));


	}
	public static void  test8()
	{
		// + - / * % all are the same operation. division by zero error not handled.
		System.out.println("========== Test 8.0 basic math operations. ==========");

		Base d1=new Base(new float[]{1,2,3}, new int[]{3});
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});

		System.out.println("------ data 1 ------");
		d1.printArray();
		System.out.println("------ data 2 ------");
		d2.printArray();
		System.out.println(getString("-", 20));

		int[] sh=getCommonShape(d1.shape, d2.shape);
		System.out.println("expected shape :" + Arrays.toString(sh));

		Base add=new Base(sh);
		int len=add.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			add.setRaw(i, op1 + op2);
		}
		System.out.println("======= add =======");
		System.out.println(add);
		add.printArray();

		Base div = new Base(sh);
		len = div.length; // length of the array.
		tmpSh = new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			div.setRaw(i, op1 / op2);
		}
		System.out.println("======= div =======");
		System.out.println(div);
		div.printArray();

		Base mod= new Base(sh);
		len = mod.length; // length of the array.
		tmpSh = new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			mod.setRaw(i, op1 % op2);
		}
		System.out.println("======= mod =======");
		System.out.println(mod);
		mod.printArray();

	}
	public static void  test7()
	{
		System.out.println("========== Test 7.0 transpose, copy, reshape and print tests ==========");
		System.out.println(" print test");
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		d2.printArray();
		System.out.println(getString("-", 20));

		System.out.println("copy test");
		Base d3=d2.copy();
		System.out.println(d3);
		d3.printArray();
		System.out.println(getString("-", 20));

		System.out.println("transpose test");
		Base d4=d2.transpose();
		System.out.println(d4);
		d4.printArray();
		System.out.println(getString("-", 20));

		System.out.println("copied transposed");
		Base d5=d4.copy();
		System.out.println(d5);
		d5.printArray();
		System.out.println(getString("-", 20));

		System.out.println("reshaped array 1");
		d4 = d4.reshape(2, 3);
		System.out.println(d4);
		d4.printArray();
		System.out.println(getString("-", 20));

		System.out.println("reshaped array 1");
		d4 = d4.reshape(-1, 2);
		System.out.println(d4);
		d4.printArray();
		System.out.println(getString("-", 20));

		System.out.println("localReshaped array 1");
		d4.reshapeLocal(2, 3);
		System.out.println(d4);
		d4.printArray();
		System.out.println(getString("-", 20));

		System.out.println("trimmed array 1");
		d4.reshapeLocal(1, 1, 1, 2, 3);
		System.out.println(d4);
		System.out.println("trimmed to");
		Base d6 = d4.trim();
		println(d6.strides, d4.strides);
		System.out.println(d6);
		System.out.println("shape change happen due to trim = " + (d4 != d6));
		System.out.println(getString("-", 20));


	}
	public static void  test6()
	{
		System.out.println("========== Test 6.0 index to shape array test ==========");
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		System.out.println(d2);
		System.out.println(getString("-", 20));

		for (int i=0;i < d2.length;i++)
		{
			int[] shp=d2.indexToShape(i);
			System.out.println("== " + Arrays.toString(shp) + " = " + d2.get(shp));
		}
		System.out.println(getString("-", 20));

		d2 = d2.reshape(3, 1, 2);
		for (int i=0;i < d2.length;i++)
		{
			int[] shp=d2.indexToShape(i);
			System.out.println("== " + Arrays.toString(shp) + " = " + d2.get(shp));
		}
		System.out.println(getString("-", 20));

	}
	public static void  test5()
	{
		System.out.println("========== Test 5.0 Array reshape test ==========");
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		System.out.println(d2);
		System.out.println(getString("-", 20));

		d2 = d2.reshape(3, 2);
		System.out.println(d2);
		System.out.println(getString("-", 20));

		System.out.println("the same");
		d2 = d2.reshape(-1, 2);
		System.out.println(d2);
		System.out.println(getString("-", 20));

		d2 = d2.reshape(2, 1, 3);
		System.out.println(d2);

		System.out.println(d2.get(10, 1, 0, 2));
		System.out.println(getString("-", 20));
		System.out.println(Arrays.toString(d2.get(new int[]{0}, 30)));

	}
	public static void  test4()
	{
		System.out.println("========== Test 4.0 single value set test ==========");
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		System.out.println(d2);
		System.out.println(getString("-", 20));

		d2.set(ar(10, 1), 73); // add 73 into data.
		System.out.println(Arrays.toString(d2.get(new int[]{0}, 30)));

//		d2.set(ar(0, 0), 88, 100); // add 88 100 times into data.
//		System.out.println(Arrays.toString(d2.get(new int[]{0}, 30)));
//
//		d2.set(ar(0, 0), 77, -1); // add 88 (-1) the remaining length times into data.
//		System.out.println(Arrays.toString(d2.get(new int[]{0}, -1)));
	}
	public static void  test3()
	{
		System.out.println("========== Test 3.0 Array value set test ==========");
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		System.out.println(d2);
		System.out.println(getString("-", 20));

		// d2.set(ar(0), 73, 43, 57, 43, 78, 58, 22); // add [73, 43, ..., 22] array into the data.

		System.out.println(Arrays.toString(d2.get(new int[]{0}, 30)));

	}
	public static void  test2()
	{
		System.out.println("========== Test 2.0 Array value get test ==========");
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		System.out.println(d2);
		System.out.println(getString("-", 20));

		System.out.println(Arrays.toString(d2.get(new int[]{0}, 10)));
		System.out.println(Arrays.toString(d2.get(new int[]{0}, -1)));
	}
	public static void  test1()
	{
		System.out.println("========== Test 1.0 single value get test ==========");
		Base d1=new Base(new float[]{1,2,3}, new int[]{1,3});
		Base d2=new Base(new float[]{1,2,3,4,5,6}, new int[]{2,3});
		System.out.println(d1);
		System.out.println(getString("-", 20));
		System.out.println(d2);
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d1.get(i));
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d1.get(0, i));
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d1.get(5, i));
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d1.get(100, 1000, i));
		System.out.println(getString("-", 50));

		for (int i=0;i < 3;i++)
			System.out.println(d2.get(i));
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d2.get(0, i));
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d2.get(5, i));
		System.out.println(getString("-", 20));

		for (int i=0;i < 3;i++)
			System.out.println(d2.get(100, 1000, i));
		System.out.println(getString("-", 20));


		System.out.println("is " + Arrays.toString(d1.shape) + " broadcastable to " + Arrays.toString(d2.shape) + " : " + d1.isBrodcastable(d2.shape));
		System.out.println("new common shape : " + Arrays.toString(getCommonShape(d1.shape, d2.shape)));


	}
}
