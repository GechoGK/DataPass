package gss.DataLoader;

import gss.arr.*;
import java.io.*;

import static gss.Util.*;

public class MNISTLoader
{
	private DataInputStream idis,ldis;
	private int numImages,numLabels,imgWidth,imgHeight;
	private int loadedImgs,loadedLbls;
	public Base loadedImages,loadedLabels;

	public MNISTLoader(String images, String labels)
	{
		init(images, labels);
	}
	private void init(String imgP, String lblP)
	{
		try
		{
			idis = new DataInputStream(new FileInputStream(imgP));
			ldis = new DataInputStream(new FileInputStream(lblP));
			idis.readInt(); // skip magic number.
			ldis.readInt(); // skip magic number.

			numImages = idis.readInt(); // number of images.
			imgWidth = idis.readInt(); // image width.
			imgHeight = idis.readInt(); // image height.
			numLabels = ldis.readInt(); // number of labels.

			if (numImages != numLabels)
				warn("the number of images is not equals to the number of labels. becareful.");

		}
		catch (Exception e)
		{
			if (idis == null)
				error("images path not found ");
			if (ldis == null)
				error("labels path not found");
		}
	}
	public void load(int count) throws IOException
	{
		loadImages(count);
		loadLabels(count);
	}
	public Base loadImages(int imgCount) throws IOException
	{
		if (imgCount == -1)
			imgCount = idis.available() / (imgWidth * imgHeight);
		if (imgCount >= idis.available() / (imgWidth * imgHeight))
			warn("image count set to max");

		float[] o=new float[imgCount * imgWidth * imgHeight];
		byte[] b=new byte[o.length];
		idis.read(b);
		for (int i=0;i < o.length;i++)
			o[i] = b[i];
		b = null;
		loadedImages = NDArray.wrap(o, imgCount, imgWidth, imgHeight);
		loadedImgs = imgCount;
		return loadedImages;
	}
	public Base loadLabels(int lblCount) throws IOException
	{
		if (lblCount == -1)
			lblCount = ldis.available();
		if (lblCount >= idis.available())
			warn("labels count set to max");

		float[] o=new float[lblCount];
		byte[] b=new byte[o.length];
		ldis.read(b);
		for (int i=0;i < o.length;i++)
			o[i] = b[i];
		b = null;
		loadedLabels = NDArray.wrap(o, lblCount);
		loadedLbls = lblCount;
		return loadedLabels;
	}
	public Base getImages(int...sh)
	{
		return loadedImages.slice(sh);
	}
	public Base getLabels(int...sh)
	{
		return loadedLabels.slice(sh);
	}
	public String getInfo()
	{
		return
			decString("MNIST image loader.", 5) + "\n" +
			"expected number of images :" + numImages + " :: " + loadedImgs + "\n" +
			"expected number of labels :" + numLabels + ":: " + loadedLbls + "\n" +
			"image size (" + imgWidth + " x " + imgHeight + ")";
	}
}
